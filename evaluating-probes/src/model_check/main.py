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

def load_hf_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def print_top_logits(logits, tokenizer, topk=10):
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

def run_model_check(config):
    for check in config['model_check']:
        print(f"---\nChecking: {check['name']}")
        ds_name = check['check_on']
        prompt_template = check['check_prompt']
        model_name = check['hf_model_name']
        run_name = config.get("run_name", "run")
        device = config.get("device")
        batch_size = check.get('batch_size', 2)  # Default batch size
        seed = config.get("seed")  # Use the same seed as the main run
        print(f"Loading HuggingFace model:", model_name)
        model, tokenizer = load_hf_model_and_tokenizer(model_name)

        # Use the same dataset creation method as in evaluation to ensure consistency
        ds = Dataset(ds_name, model=model, device=device, seed=seed)
        ds = Dataset.build_imbalanced_train_balanced_eval(ds, val_size=0.10, test_size=0.15, seed=seed)
        X_test, y_test = ds.get_test_set() # only need test set
        
        # Add debugging information
        print(f"Dataset total size: {len(ds.df)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Test set shape: {y_test.shape if hasattr(y_test, 'shape') else 'no shape'}")
        print(f"Test set type: {type(y_test)}")
        
        # Validate that we have the expected test set size and class distribution
        if len(X_test) == 0:
            raise ValueError("Test set is empty! This indicates an issue with the data splitting.")
        
        unique_labels, counts = np.unique(y_test, return_counts=True)
        print(f"Test set class distribution: {dict(zip(unique_labels, counts))}")
        
        # Get class names from config
        class_names = check.get('class_names')
        if class_names is None:
            raise ValueError("You must specify class_names in your model_check config for generalization.")
        
        # Extract class token IDs directly
        class_token_ids = {}
        for idx, name in class_names.items():
            # Encode the class name with no space (as it would appear in model response)
            token_id = tokenizer.encode(f"{name}", add_special_tokens=False)[0]
            class_token_ids[int(idx)] = token_id
        
        print(f"Extracted class token IDs: {class_token_ids}")
        for idx, token_id in class_token_ids.items():
            token = tokenizer.decode([token_id])
            print(f"  Class {idx} ({class_names[idx]}): token_id={token_id}, token='{token}'")

        # Create messages for each test prompt
        messages_list = []
        for prompt in X_test:
            formatted_prompt = prompt_template.format(prompt=prompt)
            messages = [{"role": "user", "content": formatted_prompt}]
            messages_list.append(messages)

        print(f"Running model over all {len(messages_list)} prompts...")
        
        # Save CSV path
        plot_dir = Path(f"results/{run_name}/runthrough_{ds_name}")
        plot_dir.mkdir(parents=True, exist_ok=True)
        csv_path = plot_dir / f"logit_diff_{check['name']}_model_check.csv"
        
        csv_rows = []
        
        # Process each message using apply_chat_template and extract logits
        for i, messages in tqdm(enumerate(messages_list)):
            # print(f"Processing prompt {i+1}/{len(messages_list)}")
            
            # Apply chat template and get logits
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
            if torch.cuda.is_available():
                input_ids = {k: v.cuda() for k, v in input_ids.items()}
            
            with torch.no_grad():
                outputs = model(**input_ids)
                logits = outputs.logits  # (1, seq, vocab)
            
            # Get logits for the last token (where the model would predict the next token)
            response_logits = logits[0, -1]  # Remove batch dim
            
            # Extract logits for each class
            logit_values = {}
            for idx, token_id in class_token_ids.items():
                logit_values[idx] = response_logits[token_id].item()
            
            # Compute log probabilities
            log_probs = torch.log_softmax(response_logits, dim=-1)
            log_prob_values = {}
            for idx, token_id in class_token_ids.items():
                log_prob_values[idx] = log_probs[token_id].item()
            
            # Debug: Print top 5 tokens in the logit distribution
            values, indices = torch.topk(response_logits, 5)
            # print(f"\nTop 5 tokens for prompt {i+1}:")
            # for j, (value, index) in enumerate(zip(values.tolist(), indices.tolist())):
            #     token = tokenizer.decode([index])
            #     log_prob = log_probs[index].item()
            #     print(f"  {j+1}. Token '{token}' (ID: {index}): logit={value:.3f}, log_prob={log_prob:.3f}")
            
            # # Also show the class token logits specifically
            # print("Class token logits:")
            # for idx, token_id in class_token_ids.items():
            #     token = tokenizer.decode([token_id])
            #     logit_val = logit_values[idx]
            #     log_prob_val = log_prob_values[idx]
            #     print(f"  Class {idx} ('{token}'): logit={logit_val:.3f}, log_prob={log_prob_val:.3f}")
            # print("-" * 50)
            
            # Create CSV row
            row = {
                "prompt": X_test[i], 
                "label": y_test[i]
            }
            
            # Add logits
            for idx, val in logit_values.items():
                row[f"logit_{class_names[idx]}"] = val
            
            # Add log probabilities
            for idx, val in log_prob_values.items():
                row[f"logprob_{class_names[idx]}"] = val
            
            # Add logit difference (assuming binary classification with classes 0 and 1)
            if len(logit_values) == 2:
                row["logit_diff"] = logit_values[0] - logit_values[1]
            
            csv_rows.append(row)
        
        # Save to CSV
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Saved CSV of logit diffs to: {csv_path}")
        
        # Free model and clear CUDA memory after each check
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
