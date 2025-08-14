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
from src.visualize.utils_viz import *
import os
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(1)


def load_hf_model_and_tokenizer(
    model_name,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
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
    values, indices = torch.topk(last_token_logits, topk)
    tokens = [tokenizer.decode([ix]) for ix in indices.tolist()]
    print("Top logits for last token:")
    for t, v in zip(tokens, values.tolist()):
        print(f"  {t!r} ({v:.3f})")
    print("")


def run_single_model_check(
    check,
    ds_name,
    model,
    tokenizer,
    config,
):
    """Run model check on a single dataset. Only default to values if they might not be provided."""
    prompt_template = check['check_prompt']
    few_shot_prompt = check.get('few_shot_prompt', prompt_template)  # Fallback to check_prompt if not provided
    method = check.get('method')
    run_name = config.get("run_name")
    device = config.get("device")
    batch_size = check.get('batch_size', 8)  # Default batch size
    seed = config.get("seed")  # Use the same seed as the main run
    num_tokens_to_generate = check.get('num_tokens_to_generate', 2)  # Default to 5 tokens

    # Use the full dataset (not just test set) to ensure consistency across seeds
    # We'll align by prompts later during filtering
    ds = Dataset(ds_name, model=model, device=device, seed=seed)
    X_full, y_full = ds.X, ds.y  # Get the full dataset

    # Add debugging information
    # print(f"Dataset total size: {len(ds.df)}")
    # print(f"Test set size: {len(X_test)}")
    # print(f"Test set shape: {y_test.shape if hasattr(y_test, 'shape') else 'no shape'}")
    # print(f"Test set type: {type(y_test)}")

    # Validate that we have the expected dataset size and class distribution
    if len(X_full) == 0:
        raise ValueError("Dataset is empty! This indicates an issue with the data loading.")

    unique_labels, counts = np.unique(y_full, return_counts=True)
    # print(f"Full dataset class distribution: {dict(zip(unique_labels, counts))}")

    # Get class names from config
    class_names = check.get('class_names')
    if class_names is None:
        raise ValueError("You must specify class_names in your model_check config for generalization.")

    # Extract class token IDs - support both single strings and arrays of strings
    class_token_ids = {}
    class_name_mapping = {}  # Keep track of which name was chosen for each class

    for idx, name_or_names in class_names.items():
        idx = int(idx)

        # Convert to list if it's a single string
        if isinstance(name_or_names, str):
            name_list = [name_or_names]
        elif isinstance(name_or_names, list):
            name_list = name_or_names
        else:
            raise ValueError(f"Class names must be strings or lists of strings, got {type(name_or_names)}")

        # Get token IDs for all names in this class
        token_ids_for_class = []
        for name in name_list:
            try:
                token_id = tokenizer.encode(f"{name}", add_special_tokens=False)[0]
                token_ids_for_class.append((token_id, name))
            except IndexError:
                print(f"Warning: Could not encode '{name}' for class {idx}")
                continue

        if not token_ids_for_class:
            raise ValueError(f"No valid tokens found for class {idx}")

        # Store all token IDs for this class (we'll choose the best one per prompt)
        class_token_ids[idx] = token_ids_for_class

    print(f"Extracted class token IDs:")
    for idx, token_ids_list in class_token_ids.items():
        print(f"  Class {idx}:")
        for token_id, name in token_ids_list:
            token = tokenizer.decode([token_id])
            print(f"    '{name}' -> token_id={token_id}, token='{token}'")

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

    print(f"Running model over all {len(messages_list)} prompts...")
    print(f"Looking at log probabilities across the next {num_tokens_to_generate} tokens for each class...")

    # Save CSV path - include dataset name in filename for easy matching
    plot_dir = Path(f"results/{run_name}/runthrough_{ds_name}")
    csv_path = plot_dir / f"logit_diff_{check['name']}_{ds_name}_model_check.csv"

    csv_rows = []

    # Process each message using apply_chat_template and extract logits
    for i, messages in tqdm(enumerate(messages_list)):
        # print(f"Processing prompt {i+1}/{len(messages_list)}")
        if method == "it":
            # Use chat template method
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True
                )
                # print(f"Tokenized prompt after chat template (full): {tokenizer.decode(input_ids['input_ids'][0])}")
            except ValueError as e:
                if "chat_template" in str(e):
                    # Fallback to direct tokenization
                    formatted_prompt = messages[0]["content"]
                    input_ids = tokenizer(formatted_prompt, return_tensors="pt")
                    # print(f"Tokenized prompt directly (full): {tokenizer.decode(input_ids['input_ids'][0])}")
                else:
                    raise e
        elif method == "no-it":
            # Use few-shot method (no chat template)
            formatted_prompt = messages  # messages is already the raw text
            input_ids = tokenizer(formatted_prompt, return_tensors="pt")
            # print(f"Tokenized few-shot prompt (full): {tokenizer.decode(input_ids['input_ids'][0])}")
        else:
            raise ValueError(f"Unknown model check method: {method}")

        if torch.cuda.is_available():
            input_ids = {k: v.cuda() for k, v in input_ids.items()}

        # Generate next tokens to look at their log probabilities
        generated_logits = []

        with torch.no_grad():
            # Get initial logits
            outputs = model(**input_ids)
            logits = outputs.logits  # (1, seq, vocab)
            generated_logits.append(logits[0, -1])  # Last token logits

            # Generate next tokens one by one (optimized)
            current_input_ids = input_ids['input_ids'].clone()
            attention_mask = input_ids.get('attention_mask', None)

            for _ in range(num_tokens_to_generate - 1):
                # Get logits for the next token
                outputs = model(input_ids=current_input_ids)
                next_token_logits = outputs.logits[0, -1]  # (vocab_size,)
                generated_logits.append(next_token_logits)

                # Sample the next token (greedy decoding)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                # Ensure next_token has the same number of dimensions as current_input_ids
                if current_input_ids.dim() == 2 and next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)  # Add batch dimension
                current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)

        # Stack all logits: (num_tokens, vocab_size)
        all_logits = torch.stack(generated_logits)

        # Print the 5 initial generated tokens
        generated_tokens = []
        for pos in range(min(5, num_tokens_to_generate)):
            if pos < len(generated_logits):
                next_token = torch.argmax(generated_logits[pos], dim=-1)
                token_text = tokenizer.decode([next_token])
                generated_tokens.append(token_text)
        # print(f"First 5 generated tokens: {generated_tokens}")

        # Compute log probabilities for all positions
        all_log_probs = torch.log_softmax(all_logits, dim=-1)  # (num_tokens, vocab_size)

        # Extract logits and log probabilities for each class, choosing the best token from each class array
        # across all positions
        logit_values = {}
        log_prob_values = {}
        chosen_tokens = {}  # Keep track of which token was chosen for each class
        chosen_positions = {}  # Keep track of which position had the best log prob

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

        # Debug: Print top 5 tokens in the logit distribution (for the first position)
        values, indices = torch.topk(all_logits[0], 5)
        # print(f"\n=== Prompt {i+1}/{len(messages_list)} ===")
        # print(f"Prompt: {X_full[i][:100]}...")
        # print(f"True label: {y_full[i]}")

        # print(f"\nTop 5 tokens for position 0:")
        for j, (value, index) in enumerate(zip(values.tolist(), indices.tolist())):
            token = tokenizer.decode([index])
            log_prob = all_log_probs[0, index].item()
            print(f"  {j+1}. Token '{token}' (ID: {index}): logit={value:.3f}, log_prob={log_prob:.3f}")

        # Show the class token logits specifically
        # print(f"\nClass token analysis (best across all {num_tokens_to_generate} positions):")
        for idx in class_token_ids.keys():
            logit_val = logit_values[idx]
            log_prob_val = log_prob_values[idx]
            chosen_name = chosen_tokens[idx]
            chosen_pos = chosen_positions[idx]
            # print(f"  Class {idx} (chose '{chosen_name}' at position {chosen_pos}): logit={logit_val:.3f}, log_prob={log_prob_val:.3f}")

        # Show logit difference for binary classification
        if len(logit_values) == 2:
            logit_diff = logit_values[0] - logit_values[1]
            print(f"  Logit difference (Class 0 - Class 1): {logit_diff:.3f}")
            predicted_class = 0 if logit_diff > 0 else 1
            correct = predicted_class == y_full[i]
            # print(f"  Predicted: Class {predicted_class}, Correct: {correct}")

        # print("-" * 80)

        # Create CSV row
        row = {"prompt": X_full[i], "label": y_full[i]}

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
            # For class 0 samples: don't use if logit_diff >= 0, use otherwise
            # For class 1 samples: don't use if logit_diff <= 0, use otherwise
            true_class = y_full[i]
            if true_class == 0:
                # Class 0: use if logit_diff < 0 (model correctly predicts class 0)
                use_in_filtered = 1 if logit_diff < 0 else 0
            elif true_class == 1:
                # Class 1: use if logit_diff > 0 (model correctly predicts class 1)
                use_in_filtered = 1 if logit_diff > 0 else 0
            else:
                # Fallback for any other class
                use_in_filtered = 0

            row["use_in_filtered_scoring"] = use_in_filtered

        csv_rows.append(row)

    # Save to CSV
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV of logit diffs to: {csv_path}")

    # Print summary statistics
    if len(logit_values) == 2:  # Binary classification
        correct_predictions = 0
        total_predictions = len(df)

        for _, row in df.iterrows():
            logit_diff = row["logit_diff"]
            predicted_class = 0 if logit_diff > 0 else 1
            if predicted_class == row["label"]:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total predictions: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        # Show average logit differences by class
        print(f"\nAverage logit differences by true class:")
        for class_idx in [0, 1]:
            class_data = df[df["label"] == class_idx]
            if len(class_data) > 0:
                avg_logit_diff = class_data["logit_diff"].mean()
                print(f"  Class {class_idx}: {avg_logit_diff:.3f}")


def run_model_check(
    config,
):
    for check in config['model_check']:
        print(f"---\nChecking: {check['name']}")

        # Support both single dataset and list of datasets
        check_on = check['check_on']
        if isinstance(check_on, list):
            datasets_to_check = check_on
        else:
            datasets_to_check = [check_on] if check_on else []

        prompt_template = check['check_prompt']
        model_name = check['hf_model_name']
        run_name = config.get("run_name", "run")
        device = config.get("device")
        batch_size = check.get('batch_size', 2)  # Default batch size
        seed = config.get("seed")  # Use the same seed as the main run
        num_tokens_to_generate = check.get('num_tokens_to_generate', 1)  # Default to 5 tokens
        print(f"Loading HuggingFace model:", model_name)
        model, tokenizer = load_hf_model_and_tokenizer(model_name)

        # Run the check for each dataset
        for ds_name in datasets_to_check:
            print(f"\n=== Running model check on dataset: {ds_name} ===")
            run_single_model_check(check, ds_name, model, tokenizer, config)

        # Free model and clear CUDA memory after each check
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
