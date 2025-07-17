import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import Dataset
from src.visualize.utils_viz import *
import os 
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base name for YAML config file (without _config.yaml)")
    parser.add_argument("-n", "--n_samples", type=int, default=10, help="Number of sample prompts to run")
    return parser.parse_args()

def load_yaml_config(config_base):
    yaml_path = Path(f"configs/{config_base}_config.yaml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

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

def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    for check in config['model_check']:
        print(f"---\nChecking: {check['name']}")
        ds_name = check['check_on']
        prompt_template = check['check_prompt']
        model_name = check['hf_model_name']
        run_name = config.get("run_name", "run")
        device = config.get("device")
        print("Loading HuggingFace model:", model_name)
        model, tokenizer = load_hf_model_and_tokenizer(model_name)

        ds = Dataset(ds_name, model=model, device=device)
        X_test, y_test = ds.get_test_set()
        check_prompts = prepare_check_prompts(X_test, prompt_template)
        max_len = recompute_max_len(check_prompts)

        print(f"Running model over all {len(check_prompts)} prompts...")
        all_top_logits = []
        logit_diffs = []
        csv_rows = []

        correct_tokens = {0: "male", 1: "female"}

        for orig_prompt, prompt, label in tqdm(zip(X_test, check_prompts, y_test)):
            inputs = tokenizer(prompt, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Only last token's logits
            last_token_logits = logits[0, -1]
            values, indices = torch.topk(last_token_logits, 10)
            tokens = [tokenizer.decode([ix]).strip() for ix in indices.tolist()]
            logit_pairs = list(zip(tokens, values.tolist()))
            all_top_logits.append(logit_pairs)
            # print(all_top_logits)

            # Sum logits for correct/incorrect class token in top 10 only
            correct_class = correct_tokens[label]
            incorrect_class = correct_tokens[1 - label]
            sum_correct = sum(v for t, v in logit_pairs if t.strip().lower() == correct_class)
            sum_incorrect = sum(v for t, v in logit_pairs if t.strip().lower() == incorrect_class)
            logit_diff = sum_correct - sum_incorrect
            logit_diffs.append(logit_diff)
            # Save original prompt, logit_diff, and label
            csv_rows.append({
                "prompt": orig_prompt,
                "logit_diff": logit_diff,
                "label": label
            })

        # Visualization
        plot_dir = Path(f"results/{run_name}/runthrough_{ds_name}")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"logit_hist_{check['name']}_model_check.png"
        plot_class_logit_distributions(
            all_top_logits,
            all_labels=y_test,
            correct_token_for_class=correct_tokens,
            run_name=run_name,
            save_path=plot_path
        )

        # Save CSV
        csv_path = plot_dir / f"logit_diff_{check['name']}_model_check.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Saved CSV of logit diffs to: {csv_path}")

if __name__ == "__main__":
    main()
