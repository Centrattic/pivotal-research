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

def prepare_check_prompts(original_prompts, prompt_template):
    """Return new prompts: template with {prompt} replaced by each original."""
    return [prompt_template.format(prompt=p) for p in original_prompts]

def recompute_max_len(prompts):
    return max([len(p) for p in prompts])

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
        
        check_prompts = prepare_check_prompts(X_test, prompt_template)
        max_len = recompute_max_len(check_prompts)

        print(f"Running model over all {len(check_prompts)} prompts with batch size {batch_size}...")
        logit_dicts = []
        csv_rows = []

        class_names = check.get('class_names')
        if class_names is None:
            raise ValueError("You must specify class_names in your model_check config for generalization.")
        # Use class_names values as the tokens to extract, with a space prepended
        class_token_ids = {int(idx): tokenizer.encode(f" {name}", add_special_tokens=False)[0] for idx, name in class_names.items()}

        # Visualization
        plot_dir = Path(f"results/{run_name}/runthrough_{ds_name}")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"logit_hist_{check['name']}_model_check.png"

        # Save CSV path
        csv_path = plot_dir / f"logit_diff_{check['name']}_model_check.csv"
        all_top_logits = []
        all_labels = []
        # If CSV exists, skip model run and just plot
        if csv_path.exists():
            print(f"CSV {csv_path} already exists. Skipping model run and using existing CSV for plots.")
            df = pd.read_csv(csv_path)
            all_labels = df['label'].tolist()
            
            # Validate that the CSV matches our current test set
            if len(all_labels) != len(y_test):
                raise ValueError(f"CSV has {len(all_labels)} labels but test set has {len(y_test)} examples. Dataset mismatch detected!")
            
            for _, row in df.iterrows():
                logit_dict = {}
                for idx, name in class_names.items():
                    col = f"logit_{name}"
                    if col in row:
                        logit_dict[idx] = row[col]
                all_top_logits.append(logit_dict)
        else:
            for i in range(0, len(X_test), batch_size):
                batch_prompts = check_prompts[i:i+batch_size]
                batch_labels = y_test[i:i+batch_size]
                inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits  # (batch, seq, vocab)
                # For each prompt in batch
                for j in range(len(batch_prompts)):
                    last_token_logits = logits[j, -1]
                    logit_values = {idx: last_token_logits[token_id].item() for idx, token_id in class_token_ids.items()}
                    all_top_logits.append(logit_values)
                    row = {"prompt": X_test[i+j], "label": batch_labels[j]}
                    for idx, val in logit_values.items():
                        row[f"logit_{class_names[idx]}"] = val
                    if len(logit_values) == 2:
                        row["logit_diff"] = logit_values[0] - logit_values[1]
                    csv_rows.append(row)
            all_labels = y_test.tolist()
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
            print(f"Saved CSV of logit diffs to: {csv_path}")


        # # Generalized logit diff histogram
        # diff_hist_path = plot_dir / f"logit_diff_hist_{check['name']}_model_check.png"
        # plot_logit_diffs_by_class(
        #     diff_file=str(csv_path),
        #     class_names=class_names,
        #     save_path=str(diff_hist_path)
        # )

        # class_logit_path = plot_dir / f"logit_hist_{check['name']}_model_check.png"
        # plot_class_logit_distributions(
        #     all_top_logits=all_top_logits,
        #     all_labels=all_labels,
        #     class_names=class_names,
        #     run_name=run_name,
        #     save_path=class_logit_path,
        #     bins=20,
        #     x_range=(-10, 10)
        # )
        # Free model and clear CUDA memory after each check
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
