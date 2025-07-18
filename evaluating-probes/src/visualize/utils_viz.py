import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import re

import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_logit_diffs_by_class(
    diff_file: str,
    class_names: dict,
    save_path: str = "logit_diffs_by_class_hist.png",
    main_diff: tuple = None,
    bins: int = 50,
    x_range: tuple = (-10, 10),
):
    """
    Plots histogram of logit differences for each class or for a specified pair.
    Args:
        diff_file: Path to the CSV with logit columns (e.g., logit_French, logit_English).
        class_names: dict mapping class idx to class name (e.g., {0: 'French', 1: 'English'})
        save_path: Path to save the histogram image.
        main_diff: tuple (idx1, idx2) to plot logit_class1 - logit_class2. If None and two classes, uses (0,1).
        bins: Number of bins for the histogram.
    """
    diffs = pd.read_csv(diff_file)
    class_keys = list(class_names.keys())
    if main_diff is None:
        if len(class_keys) == 2:
            main_diff = (class_keys[0], class_keys[1])
        else:
            raise ValueError("main_diff must be specified for more than 2 classes.")
    idx1, idx2 = main_diff
    name1 = class_names[idx1]
    name2 = class_names[idx2]
    logit_col1 = f"logit_{name1}"
    logit_col2 = f"logit_{name2}"
    if logit_col1 not in diffs.columns or logit_col2 not in diffs.columns:
        raise ValueError(f"Logit columns {logit_col1} and/or {logit_col2} not found in {diff_file}")
    logit_diff = diffs[logit_col1] - diffs[logit_col2]
    # Overlay by true label
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (idx, name) in enumerate(class_names.items()):
        mask = diffs['label'] == idx
        plt.hist(
            logit_diff[mask],
            bins=bins,
            range=x_range,
            alpha=0.7,
            label=f"{name} (N={mask.sum()})",
            color=color_cycle[i % len(color_cycle)],
            edgecolor="black"
        )
    plt.xlabel(f"Logit difference: {name1} - {name2}")
    plt.ylabel("Count")
    plt.title(f"Logit difference histogram: {name1} vs {name2}")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved histogram to {save_path}")
    plt.close()


def plot_class_logit_distributions(
    all_top_logits,          # list of dicts: each dict maps class idx to logit value
    all_labels,              # list of int class indices per sample
    class_names,             # dict, e.g. {0: "French", 1: "English"}
    bins=20,
    x_range=(-10, 10),
    run_name="run",
    save_path=None,
):
    """
    Plots the distribution of the logit for the correct class for each true label.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (idx, name) in enumerate(class_names.items()):
        # Collect logits for samples where the true label == idx
        class_logits = [
            logit_dict[idx]
            for logit_dict, label in zip(all_top_logits, all_labels)
            if idx in logit_dict and label == idx
        ]
        if class_logits:
            plt.hist(
                class_logits,
                bins=bins,
                range=x_range,
                alpha=0.7,
                label=f"{name} (N={len(class_logits)})",
                color=color_cycle[i % len(color_cycle)]
            )
    plt.xlabel("Logit Value for True Class")
    plt.ylabel("Frequency")
    plt.title(f"Logit Distributions for True Classes\n({run_name})")
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    diff_file = "./results/gender_experiment_gemma/runthrough_4_hist_fig_ismale/logit_diff_gender-pred_model_check.csv"
    plot_logit_diffs_by_gender(diff_file)