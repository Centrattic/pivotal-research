
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
from src.probes import LinearProbe, AttentionProbe
import glob
import os
import seaborn as sns

def plot_logit_diffs_by_class(
    probe,
    test_acts,
    test_labels,
    class_names: dict,
    save_path: str = "logit_diffs_by_class_hist.png",
    main_diff: tuple = None,
    bins: int = 50,
    x_range: tuple = (-10, 10),
):
    """
    Plots histogram of logit differences for each class or for a specified pair, recalculating scores on the fly.
    Args:
        probe: trained probe object with a .predict_logits method
        test_acts: test set activations (numpy array)
        test_labels: test set true labels (numpy array)
        class_names: dict mapping class idx to class name (e.g., {0: 'French', 1: 'English'})
        save_path: Path to save the histogram image.
        main_diff: tuple (idx1, idx2) to plot logit_class1 - logit_class2. If None and two classes, uses (0,1).
        bins: Number of bins for the histogram.
    """
    logits = probe.predict_logits(test_acts)  # shape: (n_samples, n_classes) or (n_samples, 1)
    class_keys = list(class_names.keys())
    if main_diff is None:
        if len(class_keys) == 2:
            main_diff = (class_keys[0], class_keys[1])
        else:
            raise ValueError("main_diff must be specified for more than 2 classes.")
    idx1, idx2 = main_diff
    name1 = class_names[idx1]
    name2 = class_names[idx2]
    # Handle binary and multiclass
    if logits.shape[1] == 1 or logits.ndim == 1:
        # Binary: logit for class 1 (positive class)
        # For logit diff, use logit (since only one output)
        logit_diff = logits[:, 0]
    else:
        # Multiclass: difference between two class logits
        logit_diff = logits[:, idx1] - logits[:, idx2]
    plt.figure(figsize=(8, 5))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (idx, name) in enumerate(class_names.items()):
        mask = test_labels == idx
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

def plot_rebuild_experiment_results_grid(dataclass_results_dir, probe_names, class_names, rebuild_configs, save_path=None, 
                                    metrics=('acc', 'auc', 'precision', 'recall', 'fpr'), ncols=2, y_log_scale=False, show_value_labels=False):
    """
    Plots a grid (n_probes x 2) of results from multiple probes' rebuild experiments.
    Columns: [Class Counts, Class Percents]
    Args:
        dataclass_results_dir: path to dataclass_exps_{dataset}
        probe_names: list of probe names (row labels)
        class_names: dict mapping class idx to class name
        rebuild_configs: list of dicts from config['rebuild_config']
        save_path: if provided, saves the plot to this path
        metrics: tuple/list of metric keys to plot (default: all supported)
        ncols: number of columns (default 2)
        y_log_scale: if True, use log scale for y-axis
        show_value_labels: if True, annotate each data point with its value (rounded to 3 decimals)
    """

    # Find all per-config *_results.json files in the dataclass results dir (exclude allres)
    results_json_paths = sorted(glob.glob(os.path.join(dataclass_results_dir, '*_results.json')))
    results_json_paths = [p for p in results_json_paths if '_allres' not in p]
    if not results_json_paths:
        print(f"No per-config *_results.json files found in {dataclass_results_dir}")
        return
    # Map probe_name to list of (config, result_path)
    probe_to_results = {name: [] for name in probe_names}
    for path in results_json_paths:
        for probe_name in probe_names:
            if probe_name in os.path.basename(path):
                probe_to_results[probe_name].append(path)
    n_probes = len(probe_names)
    nrows = n_probes
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    setting_titles = [
        'Class Counts',
        'Class Percents'
    ]
    # Group rebuild_configs by type
    class_counts_configs = [rc for rc in rebuild_configs if 'class_counts' in rc]
    class_percents_configs = [rc for rc in rebuild_configs if 'class_percents' in rc]
    # For each probe, plot the two settings
    for row, probe_name in enumerate(probe_names):
        # Build a mapping from config to result path
        config_to_path = {}
        for path in probe_to_results.get(probe_name, []):
            fname = os.path.basename(path)
            for rc in rebuild_configs:
                match = True
                if 'class_counts' in rc:
                    for cls in rc['class_counts']:
                        if f"class{cls}_{rc['class_counts'][cls]}" not in fname:
                            match = False
                if 'class_percents' in rc:
                    for cls in rc['class_percents']:
                        pct = int(rc['class_percents'][cls]*100)
                        if f"class{cls}_{pct}pct" not in fname:
                            match = False
                    if f"total{rc['total_samples']}" not in fname:
                        match = False
                if 'seed' in rc and f"seed{rc['seed']}" not in fname:
                    match = False
                if match:
                    config_to_path[str(rc)] = path
        # For each setting, collect data
        setting_data = [[], []]  # 0: class_counts, 1: class_percents
        for rc in class_counts_configs:
            path = config_to_path.get(str(rc))
            if not path:
                continue
            with open(path, 'r') as f:
                d = json.load(f)
            val_dict = d.get('metrics', {}).get('all_examples', d.get('metrics', {}))
            n_by_class = rc['class_counts']
            total = sum(n_by_class.values())
            pct_by_class = {cls: 100 * n_by_class[cls] / total if total > 0 else 0 for cls in n_by_class}
            setting_data[0].append((total, val_dict, n_by_class, pct_by_class))
        for rc in class_percents_configs:
            path = config_to_path.get(str(rc))
            if not path:
                continue
            with open(path, 'r') as f:
                d = json.load(f)
            val_dict = d.get('metrics', {}).get('all_examples', d.get('metrics', {}))
            n_by_class = {cls: int(rc['class_percents'][cls] * rc['total_samples']) for cls in rc['class_percents']}
            pct_by_class = {cls: rc['class_percents'][cls] * 100 for cls in rc['class_percents']}
            total = rc['total_samples']
            setting_data[1].append((total, val_dict, n_by_class, pct_by_class))
        # Sort each setting by x-axis (total samples)
        for i in range(2):
            setting_data[i].sort(key=lambda x: x[0])
        # Plot each setting in its own subplot
        for col, (data, title) in enumerate(zip(setting_data, setting_titles)):
            ax = axs[row][col]
            if data:
                x = [d[0] for d in data]
                n_by_class_list = [d[2] for d in data]
                pct_by_class_list = [d[3] for d in data]
                for metric in metrics:
                    y = [d[1].get(metric, np.nan) for d in data]
                    ax.plot(x, y, marker='o', label=metric)
                    if show_value_labels:
                        for xi, yi in zip(x, y):
                            if not np.isnan(yi):
                                ax.annotate(f"{yi:.5f}", (xi, yi), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
                # Show class counts and percents in xticklabels
                xticklabels = [f"{xi}\n" + " ".join([f"N{cls}={n_by_class[cls]}" for cls in n_by_class]) + "\n" + " ".join([f"%{cls}={pct_by_class[cls]:.1f}" for cls in pct_by_class]) for xi, n_by_class, pct_by_class in zip(x, n_by_class_list, pct_by_class_list)]
                ax.set_xticks(x)
                ax.set_xticklabels(xticklabels, rotation=30, ha='right')
                ax.legend()
            if y_log_scale:
                ax.set_yscale('log')
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{probe_name}")
            else:
                ax.set_ylabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved rebuild experiment results grid to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_probe_score_violins_from_folder(folder, class_names=None, save_path=None, max_probes=9):
    """
    For each *_results.json in the folder, loads the 'scores' and 'labels', and creates a subplot with violin plots of probe scores for each class.
    Args:
        folder: Path to directory containing *_results.json files (from evaluate_probe)
        class_names: dict mapping class index to name (optional)
        save_path: if provided, saves the plot to this path
        max_probes: maximum number of probes to plot (for grid size)
    """

    result_files = sorted(glob.glob(os.path.join(folder, '*_results.json')))
    n_probes = min(len(result_files), max_probes)
    if n_probes == 0:
        print(f"No *_results.json files found in {folder}")
        return
    ncols = min(3, n_probes)
    nrows = int(np.ceil(n_probes / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    for idx, result_path in enumerate(result_files[:max_probes]):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        with open(result_path, 'r') as f:
            d = json.load(f)
        scores = np.array(d['scores']['scores'])
        labels = np.array(d['scores']['labels'])
        if class_names is None:
            unique_classes = np.unique(labels)
            class_names_map = {c: f"Class {c}" for c in unique_classes}
        else:
            class_names_map = class_names
        data = []
        for cls in np.unique(labels):
            mask = (labels == cls)
            for s in scores[mask]:
                data.append({'score': s, 'class': class_names_map.get(cls, str(cls))})
        import pandas as pd
        df = pd.DataFrame(data)
        sns.violinplot(x='class', y='score', data=df, ax=ax, inner='box', cut=0)
        ax.set_title(os.path.basename(result_path).replace('_results.json', ''), fontsize=9)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probe Score')
    # Hide any unused subplots
    for idx in range(n_probes, nrows*ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axs[row][col])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved probe score violin subplots to {save_path}")
    else:
        plt.show()


def plot_recall_at_fpr_from_folder(folder, class_names=None, save_path=None, fpr_target=0.01, max_probes=9):
    """
    For each *_results.json in the dataclass_exps folder, loads the 'scores' and 'labels', finds the threshold where FPR ≈ fpr_target,
    and plots recall at that threshold. X-axis is the number of class 1 samples in the train set.
    Separate subplots for linear and attention probes.
    """
    from scipy.special import expit  # sigmoid

    # Only use dataclass_exps folder
    result_files = sorted(glob.glob(os.path.join(folder, '*_results.json')))
    if not result_files:
        print(f"No *_results.json files found in {folder}")
        return
    # Split by probe type
    linear_files = [f for f in result_files if 'linear' in os.path.basename(f)]
    attention_files = [f for f in result_files if 'attention' in os.path.basename(f)]
    probe_groups = [('Linear', linear_files), ('Attention', attention_files)]
    ncols = 2
    nrows = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    for col, (ptype, files) in enumerate(probe_groups):
        recalls = []
        n_class1s = []
        debug_info = []
        for f in files[:max_probes]:
            print(f"Processing file: {f}")
            # Try to extract class1 count from filename (e.g., class0_3500_class1_500)
            match = re.search(r'class1_(\d+)', f)
            n_class1 = None
            if match:
                n_class1 = int(match.group(1))
                print(f"  Extracted n_class1 from filename: {n_class1}")
            else:
                # Try to extract from meta file
                meta_file = f.replace('_results.json', '_meta.json')
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r') as mf:
                            meta = json.load(mf)
                        cc = meta.get('rebuild_config', {}).get('class_counts')
                        if cc and '1' in cc:
                            n_class1 = int(cc['1'])
                        elif cc and 1 in cc:
                            n_class1 = int(cc[1])
                        print(f"  Extracted n_class1 from meta: {n_class1}")
                    except Exception as e:
                        print(f"Warning: Could not extract class1 count from meta file: {meta_file}. Error: {e}")
            if n_class1 is None:
                print(f"Warning: Could not extract class1 count from filename or meta: {f}. Skipping.")
                continue
            with open(f, 'r') as jf:
                d = json.load(jf)
            scores = np.array(d['scores']['scores'])
            labels = np.array(d['scores']['labels'])
            # Apply sigmoid to scores for binary classification
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
            print(f"  Best recall at FPR<={fpr_target}: {best_recall:.3f} (FPR={best_fpr:.3f})")
            if best_recall > 0 or best_fpr <= fpr_target:
                n_class1s.append(n_class1)
                recalls.append(best_recall)
                debug_info.append((n_class1, best_recall, best_fpr, os.path.basename(f)))
        # Sort by n_class1
        if len(n_class1s) == 0:
            ax = axs[0][col]
            ax.set_title(f"{ptype} Probes: Recall at FPR={fpr_target}\n(No valid probes found)")
            ax.set_xlabel("Number of class 1 (positive) samples in train set")
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1)
            print(f"Warning: No valid points found for {ptype} probes in {folder}")
            continue
        n_class1s, recalls = zip(*sorted(zip(n_class1s, recalls)))
        print(f"Summary for {ptype} probes:")
        for info in sorted(debug_info):
            print(f"  n_class1={info[0]}, recall={info[1]:.3f}, fpr={info[2]:.3f}, file={info[3]}")
        ax = axs[0][col]
        ax.plot(n_class1s, recalls, 'o-', color='C0' if ptype == 'Linear' else 'C1')
        ax.set_title(f"{ptype} Probes: Recall at FPR={fpr_target}")
        ax.set_ylabel("Recall")
        ax.set_xlabel("Number of class 1 (positive) samples in train set")
        ax.set_ylim(0, 1)
        for x, y in zip(n_class1s, recalls):
            ax.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved recall@FPR plot to {save_path}")
    else:
        plt.show()

def plot_auc_vs_n_class1_from_folder(folder, class_names=None, save_path=None, max_probes=9):
    """
    For each *_results.json in the dataclass_exps folder, loads the 'scores' and 'labels', computes AUC,
    and plots AUC as a function of the number of class 1 samples in the train set.
    Separate subplots for linear and attention probes.
    """

    from scipy.special import expit  # sigmoid
    from sklearn.metrics import roc_auc_score
    result_files = sorted(glob.glob(os.path.join(folder, '*_results.json')))
    if not result_files:
        print(f"No *_results.json files found in {folder}")
        return
    linear_files = [f for f in result_files if 'linear' in os.path.basename(f)]
    attention_files = [f for f in result_files if 'attention' in os.path.basename(f)]
    probe_groups = [('Linear', linear_files), ('Attention', attention_files)]
    ncols = 2
    nrows = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    for col, (ptype, files) in enumerate(probe_groups):
        aucs = []
        n_class1s = []
        debug_info = []
        for f in files[:max_probes]:
            print(f"Processing file: {f}")
            match = re.search(r'class1_(\d+)', f)
            n_class1 = None
            if match:
                n_class1 = int(match.group(1))
                print(f"  Extracted n_class1 from filename: {n_class1}")
            else:
                meta_file = f.replace('_results.json', '_meta.json')
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r') as mf:
                            meta = json.load(mf)
                        cc = meta.get('rebuild_config', {}).get('class_counts')
                        if cc and '1' in cc:
                            n_class1 = int(cc['1'])
                        elif cc and 1 in cc:
                            n_class1 = int(cc[1])
                        print(f"  Extracted n_class1 from meta: {n_class1}")
                    except Exception as e:
                        print(f"Warning: Could not extract class1 count from meta file: {meta_file}. Error: {e}")
            if n_class1 is None:
                print(f"Warning: Could not extract class1 count from filename or meta: {f}. Skipping.")
                continue
            with open(f, 'r') as jf:
                d = json.load(jf)
            scores = np.array(d['scores']['scores'])
            labels = np.array(d['scores']['labels'])
            # Apply sigmoid to scores for binary classification
            scores = expit(scores)
            try:
                auc = roc_auc_score(labels, scores)
            except Exception as e:
                print(f"  Could not compute AUC for {f}: {e}")
                continue
            print(f"  AUC: {auc:.3f}")
            n_class1s.append(n_class1)
            aucs.append(auc)
            debug_info.append((n_class1, auc, os.path.basename(f)))
        if len(n_class1s) == 0:
            ax = axs[0][col]
            ax.set_title(f"{ptype} Probes: AUC vs. #class1\n(No valid probes found)")
            ax.set_xlabel("Number of class 1 (positive) samples in train set")
            ax.set_ylabel("AUC")
            ax.set_ylim(0, 1)
            print(f"Warning: No valid points found for {ptype} probes in {folder}")
            continue
        n_class1s, aucs = zip(*sorted(zip(n_class1s, aucs)))
        print(f"Summary for {ptype} probes (AUC):")
        for info in sorted(debug_info):
            print(f"  n_class1={info[0]}, auc={info[1]:.3f}, file={info[2]}")
        ax = axs[0][col]
        ax.plot(n_class1s, aucs, 'o-', color='C0' if ptype == 'Linear' else 'C1')
        ax.set_title(f"{ptype} Probes: AUC vs. #class1")
        ax.set_ylabel("AUC")
        ax.set_xlabel("Number of class 1 (positive) samples in train set")
        ax.set_ylim(0, 1)
        for x, y in zip(n_class1s, aucs):
            ax.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved AUC vs. #class1 plot to {save_path}")
    else:
        plt.show()
