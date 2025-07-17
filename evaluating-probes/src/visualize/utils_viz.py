import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import re

import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_logit_diffs_by_gender(
    diff_file: str, 
    gender_file: str = "./datasets/cleaned/4_hist_fig_ismale.csv",
    save_path: str = "logit_diffs_by_gender_hist.png"
):
    """
    Plots histogram of logit diffs colored by gender, matching names from prompts to gender file.

    Args:
        diff_file: Path to the CSV with prompt and logit_diff columns.
        gender_file: Path to the CSV with columns 'name' and 'is_male' (1=male, 0=female)
        save_path: Path to save the histogram image.
    """
    # Load data
    diffs = pd.read_csv(diff_file)
    gender_df = pd.read_csv(gender_file)
    gender_map = dict(zip(gender_df["prompt"], gender_df["target"]))

    # Regex to extract the person name
    def extract_name(prompt):
        m = re.match(r'^In one word, (.+?)[\’\']s gender was:', prompt)
        return m.group(1).strip() if m else None

    def strip_lower(prompt):
        return prompt.strip()

    # Extract names and look up gender
    diffs["name"] = diffs["prompt"].apply(extract_name).apply(strip_lower)
    print(diffs['name'])
    diffs["is_male"] = diffs["name"].map(gender_map)
    print(diffs["is_male"])
    diffs["gender"] = diffs["is_male"].map({True: "Male", False: "Female"})
    
    missing = diffs[diffs["is_male"].isnull()]
    if not missing.empty:
        print("Warning: Could not determine gender for names:", missing["name"].tolist())
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Male": "tab:blue", "Female": "tab:orange"}

    for gender in ["Male", "Female"]:
        sub = diffs[diffs["gender"] == gender]
        ax.hist(
            sub["logit_diff"], 
            bins=30, 
            alpha=0.7, 
            label=gender, 
            color=colors[gender],
            edgecolor="black"
        )
    
    ax.set_xlabel("Logit difference")
    ax.set_ylabel("Count")
    ax.set_title("Logit difference histogram by gender")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved histogram to {save_path}")

    plt.show()



def plot_class_logit_distributions(
    all_top_logits,          # list of [ [token(str), logit(float)], ... ] per sample
    all_labels,              # list of 0 or 1 per sample (0=male, 1=female)
    correct_token_for_class, # dict, e.g. {0: "male", 1: "female"}
    class_names = {0: "male", 1: "female"},
    bins=50,
    run_name="run",
    save_path=None,
):
    """
    all_top_logits: List of list of (token, logit) pairs per sample (top 10)
    all_labels: list of int (0 or 1) for each sample
    correct_token_for_class: dict mapping class idx to class token
    """
    # Collect all logits for correct class tokens, per class
    class_logits = {0: [], 1: []}
    for top_logits, label in zip(all_top_logits, all_labels):
        for token, logit in top_logits:
            token_str = token.strip().lower()
            for class_idx, class_token in correct_token_for_class.items():
                if token_str == class_token:
                    class_logits[class_idx].append(logit)
    
    # Plot histograms
    plt.figure(figsize=(8, 4))
    for class_idx, name in class_names.items():
        plt.hist(
            class_logits[class_idx],
            bins=bins,
            alpha=0.7,
            label=f"{name} (N={len(class_logits[class_idx])})"
        )
    plt.xlabel("Logit Value for Class Token")
    plt.ylabel("Frequency")
    plt.title(f"Logit Distributions for Class Tokens\n({run_name})")
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)

if __name__ == '__main__':
    diff_file = "./results/male_dataset_gemma/runthrough_4_hist_fig_ismale/logit_diff_male_model_check.csv"
    plot_logit_diffs_by_gender(diff_file)