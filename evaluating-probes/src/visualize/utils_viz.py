
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import glob
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import statistics

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


def _get_result_files_for_seeds(base_results_dir: Path, seeds: List[str], experiment_folder: str, 
                               architecture: str, dataclass_folder: str = None) -> Dict[str, List[str]]:
    """Get result files for each seed for a given experiment and architecture."""
    seed_files = {}
    
    for seed in seeds:
        seed_dir = base_results_dir / f"seed_{seed}" / experiment_folder
        if dataclass_folder:
            seed_dir = seed_dir / dataclass_folder
        
        if not seed_dir.exists():
            continue
            
        # Find result files for this architecture
        result_files = sorted(glob.glob(str(seed_dir / f'*{architecture}*_results.json')))
        seed_files[seed] = result_files
    
    return seed_files


def _calculate_metric_with_error_bars(seed_files: Dict[str, List[str]], metric_func, 
                                    filtered: bool = True) -> Tuple[List, List, List]:
    """Calculate metric values across seeds and return mean, std, and x_values."""
    # Group files by their class1 count across seeds
    class1_to_files = {}
    
    for seed, files in seed_files.items():
        for file in files:
            match = re.search(r'class1_(\d+)', file)
            if match:
                class1_count = int(match.group(1))
                if class1_count not in class1_to_files:
                    class1_to_files[class1_count] = []
                class1_to_files[class1_count].append(file)
    
    x_values = []
    means = []
    stds = []
    
    for class1_count in sorted(class1_to_files.keys()):
        files = class1_to_files[class1_count]
        metrics = []
        
        for file in files:
            try:
                scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                metric = metric_func(scores, labels)
                metrics.append(metric)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        if metrics:
            x_values.append(class1_count)
            means.append(np.mean(metrics))
            stds.append(np.std(metrics))
    
    return x_values, means, stds


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


def plot_multi_folder_recall_at_fpr(folders, folder_labels, architecture, class_names=None, save_path=None, 
                                   fpr_target=0.01, max_probes=20, colors=None, filtered=True, 
                                   seeds: List[str] = None):
    from scipy.special import expit
    
    if seeds is None:
        seeds = ['42']  # Default to seed 42
    
    if colors is None:
        colors = [f"C{i}" for i in range(len(folders))]
    
    plt.figure(figsize=(7, 5))
    all_recalls = []  # Collect all recall values across all folders
    
    for i, (folder, label) in enumerate(zip(folders, folder_labels)):
        # For each folder, get result files across all seeds
        # The folder path is now like: results/run_name/seed_42/2-experiment/dataclass_exps_...
        # We need to extract the experiment folder name and the dataclass folder name
        folder_path = Path(folder)
        experiment_folder = folder_path.parent.name  # e.g., "2-spam-pred-auc-increasing-spam-fixed-total"
        dataclass_folder = folder_path.name  # e.g., "dataclass_exps_94_better_spam"
        base_results_dir = folder_path.parent.parent.parent  # Go up to results/run_name
        
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, dataclass_folder)
        
        if not seed_files:
            continue
        
        def recall_at_fpr_func(scores, labels):
            scores = expit(scores)
            thresholds = np.unique(scores)[::-1]
            best_recall = 0.0
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
            return best_recall
        
        x_values, means, stds = _calculate_metric_with_error_bars(seed_files, recall_at_fpr_func, filtered)
        
        if x_values:
            all_recalls.extend(means)
            if len(seeds) > 1:
                plt.errorbar(x_values, means, yerr=stds, fmt='o-', color=colors[i], 
                           label=label, capsize=5, capthick=2)
            else:
                plt.plot(x_values, means, 'o-', color=colors[i], label=label)
            
            for x, y in zip(x_values, means):
                plt.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8, color=colors[i])
    
    plt.title(f"{architecture.capitalize()} Probes: Recall at FPR={fpr_target}" + 
              (" (filtered)" if filtered else " (all)") + 
              (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""))
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


def plot_multi_folder_auc_vs_n_class1(folders, folder_labels, architecture, class_names=None, save_path=None, 
                                     max_probes=20, colors=None, filtered=True, seeds: List[str] = None):
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']  # Default to seed 42
    
    if colors is None:
        colors = [f"C{i}" for i in range(len(folders))]
    
    plt.figure(figsize=(7, 5))
    all_aucs = []  # Collect all AUC values across all folders
    
    for i, (folder, label) in enumerate(zip(folders, folder_labels)):
        # For each folder, get result files across all seeds
        # The folder path is now like: results/run_name/seed_42/2-experiment/dataclass_exps_...
        # We need to extract the experiment folder name and the dataclass folder name
        folder_path = Path(folder)
        experiment_folder = folder_path.parent.name  # e.g., "2-spam-pred-auc-increasing-spam-fixed-total"
        dataclass_folder = folder_path.name  # e.g., "dataclass_exps_94_better_spam"
        base_results_dir = folder_path.parent.parent.parent  # Go up to results/run_name
        
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, dataclass_folder)
        
        if not seed_files:
            continue
        
        def auc_func(scores, labels):
            scores = expit(scores)
            try:
                auc = roc_auc_score(labels, scores)
                return auc
            except Exception:
                return np.nan
        
        x_values, means, stds = _calculate_metric_with_error_bars(seed_files, auc_func, filtered)
        
        if x_values:
            all_aucs.extend(means)
            if len(seeds) > 1:
                plt.errorbar(x_values, means, yerr=stds, fmt='o-', color=colors[i], 
                           label=label, capsize=5, capthick=2)
            else:
                plt.plot(x_values, means, 'o-', color=colors[i], label=label)
            
            for x, y in zip(x_values, means):
                plt.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=8, color=colors[i])
    
    plt.title(f"{architecture.capitalize()} Probes: AUC vs. #class1" + 
              (" (filtered)" if filtered else " (all)") + 
              (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""))
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


def plot_experiment_2_all_probes_with_eval(base_results_dir: Path, architecture: str, save_path=None, 
                                          filtered: bool = True, seeds: List[str] = None):
    """Plot all probes for experiment 2 with training and evaluation datasets."""
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "2-spam-pred-auc-increasing-spam-fixed-total"
    
    # Get all result files across seeds
    seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                            architecture, 'dataclass_exps_94_better_spam')
    
    if not seed_files:
        print(f"No result files found for experiment 2, architecture {architecture}")
        return
    
    # Group by class1 count and get both train and eval results
    class1_to_train_files = {}
    class1_to_eval_files = {}
    
    for seed, files in seed_files.items():
        for file in files:
            match = re.search(r'class1_(\d+)', file)
            if match:
                class1_count = int(match.group(1))
                
                # Check if this is an eval file (different dataset name)
                if 'eval_' in file or '_eval_' in file:
                    if class1_count not in class1_to_eval_files:
                        class1_to_eval_files[class1_count] = []
                    class1_to_eval_files[class1_count].append(file)
                else:
                    if class1_count not in class1_to_train_files:
                        class1_to_train_files[class1_count] = []
                    class1_to_train_files[class1_count].append(file)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training results (solid lines)
    train_x = []
    train_aucs = []
    train_stds = []
    
    for class1_count in sorted(class1_to_train_files.keys()):
        files = class1_to_train_files[class1_count]
        aucs = []
        
        for file in files:
            try:
                scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                scores = expit(scores)
                auc = roc_auc_score(labels, scores)
                aucs.append(auc)
            except Exception as e:
                print(f"Error processing train file {file}: {e}")
                continue
        
        if aucs:
            train_x.append(class1_count)
            train_aucs.append(np.mean(aucs))
            train_stds.append(np.std(aucs))
    
    # Plot evaluation results (dashed lines)
    eval_x = []
    eval_aucs = []
    eval_stds = []
    
    for class1_count in sorted(class1_to_eval_files.keys()):
        files = class1_to_eval_files[class1_count]
        aucs = []
        
        for file in files:
            try:
                scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                scores = expit(scores)
                auc = roc_auc_score(labels, scores)
                aucs.append(auc)
            except Exception as e:
                print(f"Error processing eval file {file}: {e}")
                continue
        
        if aucs:
            eval_x.append(class1_count)
            eval_aucs.append(np.mean(aucs))
            eval_stds.append(np.std(aucs))
    
    # Plot training results
    if train_x:
        if len(seeds) > 1:
            plt.errorbar(train_x, train_aucs, yerr=train_stds, fmt='o-', 
                        label=f'{architecture} Train', linewidth=2, capsize=5, capthick=2)
        else:
            plt.plot(train_x, train_aucs, 'o-', label=f'{architecture} Train', linewidth=2)
    
    # Plot evaluation results
    if eval_x:
        if len(seeds) > 1:
            plt.errorbar(eval_x, eval_aucs, yerr=eval_stds, fmt='o--', 
                        label=f'{architecture} Eval', linewidth=2, capsize=5, capthick=2)
        else:
            plt.plot(eval_x, eval_aucs, 'o--', label=f'{architecture} Eval', linewidth=2)
    
    plt.title(f"Experiment 2: {architecture.capitalize()} Probe Performance" + 
              (" (filtered)" if filtered else " (all)") + 
              (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""))
    plt.ylabel("AUC")
    plt.xlabel("Number of class 1 (positive) samples in train set")
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved experiment 2 plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_experiment_3_upsampling_heatmap(base_results_dir: Path, architecture: str, save_path=None, 
                                        filtered: bool = True, seeds: List[str] = None):
    """Plot experiment 3 as a heatmap showing upsampling ratios vs true samples."""
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "3-spam-pred-auc-llm-upsampling"
    
    # Get all result files across seeds
    seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                            architecture, 'dataclass_exps_94_better_spam')
    
    if not seed_files:
        print(f"No result files found for experiment 3, architecture {architecture}")
        return
    
    # Parse filenames to extract true samples and upsampling ratio
    # Expected format: ..._class1_X_upsample_Y_... where X is true samples, Y is upsampling ratio
    data_points = {}
    
    for seed, files in seed_files.items():
        for file in files:
            # Extract class1 count (true samples) and upsampling ratio
            class1_match = re.search(r'class1_(\d+)', file)
            upsample_match = re.search(r'upsample_(\d+)', file)
            
            if class1_match and upsample_match:
                true_samples = int(class1_match.group(1))
                upsample_ratio = int(upsample_match.group(1))
                
                key = (true_samples, upsample_ratio)
                if key not in data_points:
                    data_points[key] = []
                
                try:
                    scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                    scores = expit(scores)
                    auc = roc_auc_score(labels, scores)
                    data_points[key].append(auc)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue
    
    if not data_points:
        print("No valid data points found for experiment 3")
        return
    
    # Create heatmap data
    true_samples_list = sorted(list(set([k[0] for k in data_points.keys()])))
    upsample_ratios = sorted(list(set([k[1] for k in data_points.keys()])))
    
    heatmap_data = np.zeros((len(upsample_ratios), len(true_samples_list)))
    
    for i, ratio in enumerate(upsample_ratios):
        for j, samples in enumerate(true_samples_list):
            key = (samples, ratio)
            if key in data_points and data_points[key]:
                if len(seeds) > 1:
                    heatmap_data[i, j] = np.mean(data_points[key])
                else:
                    heatmap_data[i, j] = data_points[key][0]
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('AUC', rotation=270, labelpad=15)
    
    # Set ticks and labels
    plt.xticks(range(len(true_samples_list)), true_samples_list)
    plt.yticks(range(len(upsample_ratios)), [f"{r}x" for r in upsample_ratios])
    plt.xlabel('Number of True Samples')
    plt.ylabel('Upsampling Ratio')
    
    # Add text annotations
    for i in range(len(upsample_ratios)):
        for j in range(len(true_samples_list)):
            text = plt.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="white" if heatmap_data[i, j] < 0.5 else "black")
    
    plt.title(f"Experiment 3: {architecture.capitalize()} Probe Performance vs Upsampling" + 
              (" (filtered)" if filtered else " (all)") + 
              (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved experiment 3 heatmap to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_all_probe_loss_curves_in_folder(folder, save_path=None, max_probes=40, seeds: List[str] = None):
    import math
    
    if seeds is None:
        seeds = ['42']
    
    # For loss curves, we'll just use the first seed for now since loss curves are typically the same across seeds
    seed = seeds[0]
    
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
