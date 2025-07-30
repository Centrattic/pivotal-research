
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import glob
import os

def extract_probe_info_from_filename(filename):
    base = os.path.basename(filename)
    arch_match = re.search(r'_(linear|attention)[^_]*', base)
    layer_match = re.search(r'_L(\d+)', base)
    class0_match = re.search(r'class0_(\d+)', base)
    class1_match = re.search(r'class1_(\d+)', base)
    arch = arch_match.group(1) if arch_match else 'unknown'
    layer = layer_match.group(1) if layer_match else 'L?'
    class0 = class0_match.group(1) if class0_match else '?'
    class1 = class1_match.group(1) if class1_match else '?'
    return f"{arch}_L{layer}_c0_{class0}_c1_{class1}"


def _get_scores_and_labels_from_result_file(result_path, filtered=True):
    with open(result_path, 'r') as f:
        d = json.load(f)
    # Try to use filtered_scores if present and requested
    if filtered:
        if 'filtered_scores' in d:
            scores = np.array(d['filtered_scores']['scores'])
            labels = np.array(d['filtered_scores']['labels'])
            return scores, labels
        elif 'scores' in d and d['scores'].get('filtered', False):
            scores = np.array(d['scores']['scores'])
            labels = np.array(d['scores']['labels'])
            return scores, labels
        elif 'all_scores' in d:
            scores = np.array(d['all_scores']['scores'])
            labels = np.array(d['all_scores']['labels'])
            return scores, labels
    else:
        if 'all_scores' in d:
            scores = np.array(d['all_scores']['scores'])
            labels = np.array(d['all_scores']['labels'])
            return scores, labels
        elif 'scores' in d and not d['scores'].get('filtered', False):
            scores = np.array(d['scores']['scores'])
            labels = np.array(d['scores']['labels'])
            return scores, labels
        elif 'filtered_scores' in d:
            scores = np.array(d['filtered_scores']['scores'])
            labels = np.array(d['filtered_scores']['labels'])
            return scores, labels
    # fallback
    scores = np.array(d['scores']['scores'])
    labels = np.array(d['scores']['labels'])
    return scores, labels


def plot_logit_diffs_from_csv(csv_path, class_names, save_path=None, bins=50, x_range=(-10, 10)):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (idx, name) in enumerate(class_names.items()):
        mask = df['label'] == idx
        plt.hist(
            df.loc[mask, 'logit_diff'],
            bins=bins,
            range=x_range,
            alpha=0.7,
            label=f"{name} (N={mask.sum()})",
            color=color_cycle[i % len(color_cycle)],
            edgecolor="black"
        )
    plt.xlabel("Logit difference")
    plt.ylabel("Count")
    plt.title("Logit difference histogram from CSV")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved histogram to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_multi_folder_recall_at_fpr(folders, folder_labels, architecture, class_names=None, save_path=None, fpr_target=0.01, max_probes=20, colors=None, filtered=True):
    from scipy.special import expit
    if colors is None:
        colors = [f"C{i}" for i in range(len(folders))]
    plt.figure(figsize=(7, 5))
    all_recalls = []  # Collect all recall values across all folders
    for i, (folder, label) in enumerate(zip(folders, folder_labels)):
        result_files = sorted(glob.glob(os.path.join(folder, f'*{architecture}*_results.json')))
        n_class1s = []
        recalls = []
        for f in result_files[:max_probes]:
            match = re.search(r'class1_(\d+)', f)
            n_class1 = int(match.group(1)) if match else None
            if n_class1 is None:
                continue
            scores, labels = _get_scores_and_labels_from_result_file(f, filtered=filtered)
            scores = expit(scores)
            thresholds = np.unique(scores)[::-1]
            best_recall = 0.0
            best_fpr = 1.0
            for thresh in thresholds:
                preds = (scores >= thresh).astype(int)
                tp = np.sum((preds == 1) & (labels == 1))
                fn = np.sum((preds == 0) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                tn = np.sum((preds == 0) & (labels == 0))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if fpr <= fpr_target and recall > best_recall:
                    best_recall = recall
                    best_fpr = fpr
            if best_recall > 0 or best_fpr <= fpr_target:
                n_class1s.append(n_class1)
                recalls.append(best_recall)
                all_recalls.append(best_recall)  # Add to global collection
        if n_class1s:
            n_class1s, recalls = zip(*sorted(zip(n_class1s, recalls)))
            plt.plot(n_class1s, recalls, 'o-', color=colors[i], label=label)
            for x, y in zip(n_class1s, recalls):
                plt.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8, color=colors[i])
    plt.title(f"{architecture.capitalize()} Probes: Recall at FPR={fpr_target}" + (" (filtered)" if filtered else " (all)"))
    plt.ylabel("Recall")
    plt.xlabel("Number of class 1 (positive) samples in train set")
    plt.xscale('log')
    # Set y-axis to start at a reasonable nonzero value based on the data
    if all_recalls:
        min_recall = min(all_recalls)
        y_min = max(0.0, min_recall - 0.1)  # Start at least 0.1 below the minimum, but not below 0
        plt.ylim(y_min, 1)
    else:
        plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved multi-folder recall@FPR plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_multi_folder_auc_vs_n_class1(folders, folder_labels, architecture, class_names=None, save_path=None, max_probes=20, colors=None, filtered=True):
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    if colors is None:
        colors = [f"C{i}" for i in range(len(folders))]
    plt.figure(figsize=(7, 5))
    all_aucs = []  # Collect all AUC values across all folders
    for i, (folder, label) in enumerate(zip(folders, folder_labels)):
        result_files = sorted(glob.glob(os.path.join(folder, f'*{architecture}*_results.json')))
        n_class1s = []
        aucs = []
        for f in result_files[:max_probes]:
            match = re.search(r'class1_(\d+)', f)
            n_class1 = int(match.group(1)) if match else None
            if n_class1 is None:
                continue
            scores, labels = _get_scores_and_labels_from_result_file(f, filtered=filtered)
            scores = expit(scores)
            try:
                auc = roc_auc_score(labels, scores)
            except Exception:
                continue
            n_class1s.append(n_class1)
            aucs.append(auc)
            all_aucs.append(auc)  # Add to global collection
        if n_class1s:
            n_class1s, aucs = zip(*sorted(zip(n_class1s, aucs)))
            plt.plot(n_class1s, aucs, 'o-', color=colors[i], label=label)
            for x, y in zip(n_class1s, aucs):
                plt.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8, color=colors[i])
    plt.title(f"{architecture.capitalize()} Probes: AUC vs. #class1" + (" (filtered)" if filtered else " (all)"))
    plt.ylabel("AUC")
    plt.xlabel("Number of class 1 (positive) samples in train set")
    plt.xscale('log')
    # Set y-axis to start at a reasonable nonzero value based on the data
    if all_aucs:
        min_auc = min(all_aucs)
        y_min = max(0.0, min_auc - 0.1)  # Start at least 0.1 below the minimum, but not below 0
        plt.ylim(y_min, 1)
    else:
        plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved multi-folder AUC vs #class1 plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_all_probe_loss_curves_in_folder(folder, save_path=None, max_probes=40):
    import math
    log_files = sorted(glob.glob(os.path.join(folder, '*_train_log.json')))
    n = min(len(log_files), max_probes)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), squeeze=False)
    for idx, log_path in enumerate(log_files[:max_probes]):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        with open(log_path, 'r') as f:
            d = json.load(f)
        loss_history = d.get('loss_history', [])
        ax.plot(loss_history)
        label = extract_probe_info_from_filename(log_path)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
    # Hide unused subplots
    for idx in range(n, nrows*ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axs[row][col])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved all probe loss curves to {save_path}")
        plt.close()
    else:
        plt.show()
