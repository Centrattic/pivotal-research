import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import pandas as pd

# Clear CUDA memory at the very start
if torch.cuda.is_available():
    print("[model_check] Clearing CUDA memory at start...")
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import Dataset
import os
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(1)


def load_hf_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def print_top_logits(
    logits,
    tokenizer,
    topk=10,
):
    # logits: (seq_len, vocab_size) or (1, seq_len, vocab_size)
    if logits.dim() == 3:
        logits = logits[0]  # Remove batch dim
    last_token_logits = logits[-1]
    values, indices = torch.topk(
        last_token_logits,
        topk,
    )
    tokens = [tokenizer.decode([ix]) for ix in indices.tolist()]
    print("Top logits for last token:")
    for t, v in zip(
            tokens,
            values.tolist(),
    ):
        print(f"  {t!r} ({v:.3f})")
    print("")


def run_single_model_check(
    check,
    ds_name,
    model,
    tokenizer,
    config,
    logger,
):
    """Run model check on a single dataset. Only default to values if they might not be provided."""
    prompt_template = check['check_prompt']
    few_shot_prompt = check.get(
        'few_shot_prompt',
        prompt_template,
    )  # Fallback to check_prompt if not provided
    method = check.get('method')
    run_name = config.get("run_name")
    device = config.get("device")
    seed = config.get("seed")  # Use the same seed as the main run
    num_tokens_to_generate = check.get(
        'num_tokens_to_generate',
        1,
    )

    # Use the full dataset (not just test set) to ensure consistency across seeds
    # We'll align by prompts later during filtering
    ds = Dataset(
        ds_name,
        model=model,
        device=device,
        seed=seed,
    )
    X_full, y_full = ds.X, ds.y  # Get the full dataset

    # Add debugging information
    logger.log(f"Dataset total size from df: {len(ds.df)}")
    logger.log(f"X_full size: {len(X_full)}")
    logger.log(f"y_full size: {len(y_full)}")
    logger.log(f"Test set size: {len(ds.X_test_text) if ds.X_test_text is not None else 'Not split yet'}")
    logger.log(f"Train set size: {len(ds.X_train_text) if ds.X_train_text is not None else 'Not split yet'}")

    # Validate that we have the expected dataset size and class distribution
    if len(X_full) == 0:
        raise ValueError("Dataset is empty! This indicates an issue with the data loading.")

    unique_labels, counts = np.unique(
        y_full,
        return_counts=True,
    )
    logger.log(f"Full dataset class distribution: {dict(zip(unique_labels, counts,))}")

    # Get class names from config
    class_names = check.get('class_names')
    if class_names is None:
        raise ValueError("You must specify class_names in your model_check config for generalization.")

    # Extract class token IDs - support both single strings and arrays of strings
    class_token_ids = {}

    for idx, name_or_names in class_names.items():
        idx = int(idx)

        # Convert to list if it's a single string
        if isinstance(
                name_or_names,
                str,
        ):
            name_list = [name_or_names]
        elif isinstance(
                name_or_names,
                list,
        ):
            name_list = name_or_names
        else:
            raise ValueError(f"Class names must be strings or lists of strings, got {type(name_or_names)}")

        # Get token IDs for all names in this class
        token_ids_for_class = []
        for name in name_list:
            try:
                token_id = tokenizer.encode(
                    f"{name}",
                    add_special_tokens=False,
                )[0]
                token_ids_for_class.append((
                    token_id,
                    name,
                ))
            except IndexError:
                print(f"Warning: Could not encode '{name}' for class {idx}")
                continue

        if not token_ids_for_class:
            raise ValueError(f"No valid tokens found for class {idx}")

        # Store all token IDs for this class (we'll choose the best one per prompt)
        class_token_ids[idx] = token_ids_for_class

    logger.log(f"Extracted class token IDs:")
    for idx, token_ids_list in class_token_ids.items():
        logger.log(f"  Class {idx}:")
        for token_id, name in token_ids_list:
            token = tokenizer.decode([token_id])
            logger.log(f"    '{name}' -> token_id={token_id}, token='{token}'")

    # Create messages for each dataset example
    messages_list = []
    for prompt in X_full:
        if method == "it":
            # Use chat template method
            formatted_prompt = prompt_template.format(prompt=prompt)
            messages = [{"role": "user", "content": formatted_prompt}]
            messages_list.append(messages)
        elif method == "no-it":
            # Use few-shot method (no chat template)
            formatted_prompt = few_shot_prompt.format(prompt=prompt)
            messages_list.append(formatted_prompt)  # Just the raw text, no chat format
        else:
            raise ValueError(f"Unknown model check method: {method}. Must be 'it' or 'no-it'")

    # Save CSV path - include dataset name in filename for easy matching
    plot_dir = Path(f"results/{run_name}/runthrough_{ds_name}")
    csv_path = plot_dir / f"logit_diff_{check['name']}_{ds_name}_model_check.csv"

    csv_rows = []
    # Process messages in batches for efficiency
    processed_count = 0
    error_count = 0
    batch_size = check.get(
        'batch_size',
        2,
    )  # Default batch size for processing

    logger.log(f"Running model over all {len(messages_list)} prompts...")
    logger.log(f"Looking at log probabilities across the next {num_tokens_to_generate} tokens for each class...")
    logger.log(f"Dataset size: {len(X_full)} examples")
    logger.log(f"Class distribution: {dict(zip(unique_labels, counts,))}")
    logger.log(f"Processing in batches of size: {batch_size}")

    # Initialize logit_values outside the loop to avoid UnboundLocalError
    logit_values = {}

    # First, tokenize all prompts to get their lengths for proper batching
    logger.log("Pre-tokenizing all prompts to determine batch padding...")
    all_tokenized = []
    for messages in tqdm(
            messages_list,
            desc="Pre-tokenizing prompts",
    ):
        if method == "it":
            # Use chat template method
            try:
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
            except ValueError as e:
                if "chat_template" in str(e):
                    # Fallback to direct tokenization
                    formatted_prompt = messages[0]["content"]
                    tokenized = tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                    )
                else:
                    raise e
        elif method == "no-it":
            # Use few-shot method (no chat template)
            formatted_prompt = messages  # messages is already the raw text
            tokenized = tokenizer(
                formatted_prompt,
                return_tensors="pt",
            )
        else:
            raise ValueError(f"Unknown model check method: {method}")

        all_tokenized.append(tokenized)

    # Process in batches
    for batch_start in tqdm(
            range(
                0,
                len(messages_list),
                batch_size,
            ),
            desc=f"Processing {ds_name} examples in batches",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] ({percentage:3.0f}%)',
    ):
        batch_end = min(
            batch_start + batch_size,
            len(messages_list),
        )
        batch_messages = messages_list[batch_start:batch_end]
        batch_tokenized = all_tokenized[batch_start:batch_end]

        try:
            # Get the maximum sequence length in this batch
            max_seq_len = max(
                tokenized['input_ids'].size(
                    1) for tokenized in batch_tokenized\
            )

            # Pad all examples in the batch to the same length
            batch_input_ids = []
            batch_attention_masks = []

            for tokenized in batch_tokenized:
                seq_len = tokenized['input_ids'].size(1)
                padding_len = max_seq_len - seq_len

                if padding_len > 0:
                    # Pad input_ids
                    padded_input_ids = torch.cat(
                        [
                            tokenized['input_ids'],
                            torch.full(
                                (1, padding_len),
                                tokenizer.pad_token_id,
                                dtype=tokenized['input_ids'].dtype,
                            ),
                        ],
                        dim=1,
                    )

                    # Pad attention mask
                    if 'attention_mask' in tokenized:
                        padded_attention_mask = torch.cat(
                            [
                                tokenized['attention_mask'],
                                torch.zeros(
                                    (1, padding_len),
                                    dtype=tokenized['attention_mask'].dtype,
                                )
                            ],
                            dim=1,
                        )
                    else:
                        padded_attention_mask = torch.cat(
                            [
                                torch.ones(
                                    (1, seq_len),
                                    dtype=torch.long,
                                ),
                                torch.zeros(
                                    (1, padding_len),
                                    dtype=torch.long,
                                ),
                            ],
                            dim=1,
                        )
                else:
                    padded_input_ids = tokenized['input_ids']
                    padded_attention_mask = tokenized.get(
                        'attention_mask',
                        torch.ones(
                            (1, seq_len),
                            dtype=torch.long,
                        ),
                    )

                batch_input_ids.append(padded_input_ids)
                batch_attention_masks.append(padded_attention_mask)

            # Stack the batch
            batch_input_dict = {
                'input_ids': torch.cat(
                    batch_input_ids,
                    dim=0,
                ),
                'attention_mask': torch.cat(
                    batch_attention_masks,
                    dim=0,
                ),
            }

            if torch.cuda.is_available():
                batch_input_dict = {k: v.cuda() for k, v in batch_input_dict.items()}

            # Process the entire batch through the model
            batch_logits_list = []

            with torch.no_grad():
                # Get initial logits for the entire batch
                outputs = model(**batch_input_dict)
                batch_logits = outputs.logits  # (batch_size, seq, vocab)

                # Store initial logits for each example
                for i in range(batch_logits.size(0)):
                    batch_logits_list.append([batch_logits[i, -1]])  # Last token logits for each example

                # Generate next tokens for the entire batch
                current_input_ids = batch_input_dict['input_ids'].clone()
                current_attention_mask = batch_input_dict['attention_mask'].clone()

                for _ in range(num_tokens_to_generate - 1):
                    # Get logits for the next token for entire batch
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                    )

                    next_token_logits = outputs.logits[:, -1]  # (batch_size, vocab_size)

                    # Store logits for each example
                    for i in range(next_token_logits.size(0)):
                        batch_logits_list[i].append(next_token_logits[i])

                    # Sample next tokens (greedy decoding) for entire batch
                    next_tokens = torch.argmax(
                        next_token_logits,
                        dim=-1,
                        keepdim=True,
                    )
                    current_input_ids = torch.cat(
                        [current_input_ids, next_tokens],
                        dim=-1,
                    )

                    # Update attention mask
                    new_attention = torch.ones(
                        current_input_ids.size(0),
                        1,
                        device=current_input_ids.device,
                    )
                    current_attention_mask = torch.cat(
                        [current_attention_mask, new_attention],
                        dim=-1,
                    )

            # Process each example's results
            for i, (messages, example_logits) in enumerate(zip(
                    batch_messages,
                    batch_logits_list,
            )):
                global_idx = batch_start + i

                # Stack all logits for this example: (num_tokens, vocab_size)
                all_logits = torch.stack(example_logits)

                # Compute log probabilities for all positions
                all_log_probs = torch.log_softmax(
                    all_logits,
                    dim=-1,
                )  # (num_tokens, vocab_size)

                # Extract logits and log probabilities for each class
                logit_values = {}
                log_prob_values = {}
                chosen_tokens = {}
                chosen_positions = {}

                for idx, token_ids_list in class_token_ids.items():
                    best_log_prob = float('-inf')
                    best_token_id = None
                    best_token_name = None
                    best_logit = None
                    best_position = None

                    # Find the token with highest log probability for this class across all positions
                    for token_id, name in token_ids_list:
                        # Check log probability at each position
                        for pos in range(num_tokens_to_generate):
                            log_prob_val = all_log_probs[pos, token_id].item()
                            if log_prob_val > best_log_prob:
                                best_log_prob = log_prob_val
                                best_token_id = token_id
                                best_token_name = name
                                best_logit = all_logits[pos, token_id].item()
                                best_position = pos

                    logit_values[idx] = best_logit
                    log_prob_values[idx] = best_log_prob
                    chosen_tokens[idx] = best_token_name
                    chosen_positions[idx] = best_position

                # Debug prints: top-5 tokens at the final position and Yes/No diffs
                try:
                    final_pos = all_logits.size(0) - 1
                    final_logits = all_logits[final_pos]
                    final_log_probs = all_log_probs[final_pos]
                    topk = 5
                    top_values, top_indices = torch.topk(
                        final_logits,
                        k=topk,
                    )
                    print("\n--- Example", global_idx)
                    # Show the actual rendered prompt passed to the tokenizer/model
                    if method == "it":
                        try:
                            rendered_prompt = messages[0]["content"]
                        except Exception:
                            rendered_prompt = str(messages)
                    else:
                        rendered_prompt = messages  # already a formatted string for no-it

                    prompt_preview = str(rendered_prompt)[:120].replace(
                        "\n",
                        " ",
                    )
                    print(f"Prompt preview: {prompt_preview!r}")
                    print("Full prompt:")
                    print(rendered_prompt)
                    print("Top-5 tokens at final position:")
                    for rank, (val, idx_tok) in enumerate(
                            zip(
                                top_values.tolist(),
                                top_indices.tolist(),
                            ),
                            start=1,
                    ):
                        tok = tokenizer.decode([idx_tok])
                        lp = final_log_probs[idx_tok].item()
                        print(f"  {rank}. {tok!r}  logit={val:.3f}  logprob={lp:.3f}")

                    if 0 in logit_values and 1 in logit_values:
                        yes_name = chosen_tokens.get(
                            0,
                            "<unk>",
                        )
                        no_name = chosen_tokens.get(
                            1,
                            "<unk>",
                        )
                        yes_pos = chosen_positions.get(
                            0,
                            None,
                        )
                        no_pos = chosen_positions.get(
                            1,
                            None,
                        )
                        yes_logit = logit_values[0]
                        no_logit = logit_values[1]
                        yes_lp = log_prob_values[0]
                        no_lp = log_prob_values[1]
                        print("Best class tokens and diffs:")
                        print(f"  Yes: token={yes_name!r}  pos={yes_pos}  logit={yes_logit:.3f}  logprob={yes_lp:.3f}")
                        print(f"  No : token={no_name!r}  pos={no_pos}  logit={no_logit:.3f}  logprob={no_lp:.3f}")
                        print(f"  logit_diff (Yes - No): {yes_logit - no_logit:.3f}")
                        print(f"  logit_diff (No - Yes): {no_logit - yes_logit:.3f}")
                except Exception as dbg_e:
                    print(f"[debug-print] Skipped due to error: {dbg_e}")

                # Create CSV row
                row = {"prompt": X_full[global_idx], "label": y_full[global_idx]}

                # Add logits
                for idx, val in logit_values.items():
                    chosen_name = chosen_tokens[idx]
                    row[f"logit_{chosen_name}"] = val

                # Add log probabilities
                for idx, val in log_prob_values.items():
                    chosen_name = chosen_tokens[idx]
                    row[f"logprob_{chosen_name}"] = val

                # Add positions where best log probs were found
                for idx, pos in chosen_positions.items():
                    chosen_name = chosen_tokens[idx]
                    row[f"position_{chosen_name}"] = pos

                # Add logit difference (assuming binary classification with classes 0 and 1)
                if len(logit_values) == 2:
                    logit_diff = logit_values[0] - logit_values[1]
                    row["logit_diff"] = logit_diff

                    # Add new column for filtered scoring
                    true_class = y_full[global_idx]
                    if true_class == 0:
                        use_in_filtered = 1 if logit_diff < 0 else 0
                    elif true_class == 1:
                        use_in_filtered = 1 if logit_diff > 0 else 0
                    else:
                        use_in_filtered = 0

                    row["use_in_filtered_scoring"] = use_in_filtered

                csv_rows.append(row)
                processed_count += 1

        except Exception as e:
            error_count += len(batch_messages)
            logger.log(f"⚠️  Error processing batch starting at {batch_start}: {str(e)}")
            continue

    # Save to CSV
    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    df = pd.DataFrame(csv_rows)
    df.to_csv(
        csv_path,
        index=False,
    )
    logger.log(f"Saved CSV of logit diffs to: {csv_path}")
    logger.log(
        f"Processed {processed_count} examples successfully, {error_count} errors out of {len(X_full)} total examples"
    )

    # Verify we processed all examples
    if processed_count != len(X_full):
        logger.log(
            f"⚠️  WARNING: Only processed {processed_count} examples successfully but dataset has {len(X_full)} examples!"
        )
        if error_count > 0:
            logger.log(f"   {error_count} examples failed due to errors")
    else:
        logger.log(f"✅ Successfully processed all {processed_count} examples from the dataset")

    # Print summary statistics
    if processed_count > 0 and len(logit_values) == 2 and len(df) > 0:  # Binary classification
        total_predictions = len(df)

        # New correctness criterion:
        # - Class 0 is correct if logit_diff < 0
        # - Class 1 is correct if logit_diff > 0
        class0_mask = df["label"] == 0
        class1_mask = df["label"] == 1
        class0_total = int(class0_mask.sum())
        class1_total = int(class1_mask.sum())

        class0_correct = int((class0_mask & (df["logit_diff"] < 0)).sum())
        class1_correct = int((class1_mask & (df["logit_diff"] > 0)).sum())

        overall_correct = class0_correct + class1_correct
        overall_accuracy = overall_correct / total_predictions if total_predictions > 0 else 0.0

        logger.log(f"\n=== SUMMARY STATISTICS ===")
        logger.log(f"Total predictions: {total_predictions}")
        logger.log(f"Correct predictions (new criterion): {overall_correct}")
        logger.log(f"Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")

        # Per-class accuracy
        class0_accuracy = (class0_correct / class0_total) if class0_total > 0 else 0.0
        class1_accuracy = (class1_correct / class1_total) if class1_total > 0 else 0.0
        logger.log(f"\nPer-class accuracy:")
        logger.log(
            f"  Class 0: {class0_correct}/{class0_total} ({class0_accuracy*100:.1f}%) [criterion: logit_diff < 0]"
        )
        logger.log(
            f"  Class 1: {class1_correct}/{class1_total} ({class1_accuracy*100:.1f}%) [criterion: logit_diff > 0]"
        )

        # Show average logit differences by class
        logger.log(f"\nAverage logit differences by true class:")
        for class_idx in [0, 1]:
            class_data = df[df["label"] == class_idx]
            if len(class_data) > 0:
                avg_logit_diff = class_data["logit_diff"].mean()
                logger.log(f"  Class {class_idx}: {avg_logit_diff:.3f}")
    elif processed_count == 0:
        logger.log(f"\n⚠️  No examples were processed successfully. Cannot compute summary statistics.")


def run_model_check(
    config,
    logger,
):
    for check in config['model_check']:
        logger.log(f"---\nChecking: {check['name']}")

        # Support both single dataset and list of datasets
        check_on = check['check_on']
        if isinstance(
                check_on,
                list,
        ):
            datasets_to_check = check_on
        else:
            datasets_to_check = [check_on] if check_on else []

        model_name = check['hf_model_name']
        logger.log(f"Loading HuggingFace model: {model_name}")
        model, tokenizer = load_hf_model_and_tokenizer(model_name)

        # Run the check for each dataset
        for ds_name in datasets_to_check:
            logger.log(f"\n=== Running model check on dataset: {ds_name} ===")
            run_single_model_check(
                check,
                ds_name,
                model,
                tokenizer,
                config,
                logger=logger,
            )

        # Free model and clear CUDA memory after each check
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
