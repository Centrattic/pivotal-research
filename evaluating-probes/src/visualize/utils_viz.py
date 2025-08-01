
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
        # Check if this is experiment 3 (LLM upsampling) or experiment 2 (class-based)
        if "llm-upsampling" in experiment_folder:
            # Experiment 3: eval_on_*__*_llm_neg*_pos*_*x_seed*_*_results.json
            result_files = sorted(glob.glob(str(seed_dir / f'eval_on_*__*{architecture}*_llm_neg*_pos*_*x_seed*_*_results.json')))
        else:
            # Experiment 2: eval_on_*__*_class*_*_seed*_*_results.json
            result_files = sorted(glob.glob(str(seed_dir / f'eval_on_*__*{architecture}*_class*_*_seed*_*_results.json')))
        seed_files[seed] = result_files
    
    return seed_files


def _calculate_metric_with_error_bars(seed_files: Dict[str, List[str]], metric_func, 
                                    filtered: bool = True) -> Tuple[List, List, List]:
    """Calculate metric values across seeds and return mean, std, and x_values."""
    # Group files by their class1 count across seeds
    class1_to_files = {}
    
    for seed, files in seed_files.items():
        for file in files:
            # Updated regex to match new filename structure
            # Look for class1_X pattern in the filename
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


def plot_experiment_2_all_probes_with_eval(base_results_dir: Path, architectures: List[str], save_path=None, 
                                          filtered: bool = True, seeds: List[str] = None):
    """Plot all architectures for experiment 2 with training and evaluation datasets."""
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "2-spam-pred-auc-increasing-spam-fixed-total"
    
    plt.figure(figsize=(12, 8))
    colors = [f"C{i}" for i in range(len(architectures))]
    
    for arch_idx, architecture in enumerate(architectures):
        # Get all result files across seeds for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        
        if not seed_files:
            print(f"No result files found for experiment 2, architecture {architecture}")
            continue
        
        # Group by class1 count and get both train and eval results
        class1_to_train_files = {}
        class1_to_eval_files = {}
        
        for seed, files in seed_files.items():
            for file in files:
                match = re.search(r'class1_(\d+)', file)
                if match:
                    class1_count = int(match.group(1))
                    
                    # Check if this is an eval file (contains eval_on_ in filename)
                    if 'eval_on_' in file:
                        if class1_count not in class1_to_eval_files:
                            class1_to_eval_files[class1_count] = []
                        class1_to_eval_files[class1_count].append(file)
                    else:
                        if class1_count not in class1_to_train_files:
                            class1_to_train_files[class1_count] = []
                        class1_to_train_files[class1_count].append(file)
        
        color = colors[arch_idx]
        
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
        
        # Plot training results (solid lines)
        if train_x:
            if len(seeds) > 1:
                plt.errorbar(train_x, train_aucs, yerr=train_stds, fmt='o-', 
                            label=f'{architecture}', linewidth=2, capsize=5, capthick=2, color=color)
            else:
                plt.plot(train_x, train_aucs, 'o-', label=f'{architecture}', linewidth=2, color=color)
        
        # Plot evaluation results (dashed lines) - if any exist
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
        
        # Plot evaluation results (dashed lines in same color) - only if eval data exists
        if eval_x:
            if len(seeds) > 1:
                plt.errorbar(eval_x, eval_aucs, yerr=eval_stds, fmt='o--', 
                            label=f'{architecture} Eval', linewidth=2, capsize=5, capthick=2, color=color)
            else:
                plt.plot(eval_x, eval_aucs, 'o--', label=f'{architecture} Eval', linewidth=2, color=color)
    
    plt.title(f"Experiment 2: All Architecture Probe Performance" + 
              (" (filtered)" if filtered else " (all)") + 
              (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""))
    plt.ylabel("AUC")
    plt.xlabel("Number of class 1 (positive) samples in train set")
    plt.xscale('log')
    
    # Set y-axis to start at a reasonable nonzero value based on the data
    all_aucs = []
    for arch_idx, architecture in enumerate(architectures):
        # Collect all AUC values for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        if seed_files:
            for seed, files in seed_files.items():
                for file in files:
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        scores = expit(scores)
                        auc = roc_auc_score(labels, scores)
                        all_aucs.append(auc)
                    except Exception:
                        continue
    
    if all_aucs:
        min_auc = min(all_aucs)
        y_min = max(0.4, min_auc - 0.1)  # Start at least 0.1 below the minimum, but not below 0.4
        plt.ylim(y_min, 1)
    else:
        plt.ylim(0.4, 1)  # Fallback range
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved experiment 2 plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_experiment_3_upsampling_lineplot(base_results_dir: Path, architectures: List[str], save_path=None, 
                                         filtered: bool = True, seeds: List[str] = None):
    """Plot experiment 3 as a multi-layer line plot showing upsampling factors vs true samples for all architectures."""
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "3-spam-pred-auc-llm-upsampling"
    
    # Collect data for all architectures
    all_data = {}
    
    for architecture in architectures:
        # Get all result files across seeds for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        
        if not seed_files:
            print(f"No result files found for experiment 3, architecture {architecture}")
            continue
        
        # Parse filenames to extract true samples and upsampling ratio
        for seed, files in seed_files.items():
            for file in files:
                # Extract true samples (n_real_pos) and upsampling ratio from rebuild suffix
                llm_match = re.search(r'llm_neg(\d+)_pos(\d+)_(\d+)x', file)
                
                if llm_match:
                    n_real_neg = int(llm_match.group(1))
                    n_real_pos = int(llm_match.group(2))  # This is the true number of positive samples
                    upsampling_factor = int(llm_match.group(3))
                    
                    key = (architecture, n_real_pos, upsampling_factor)
                    if key not in all_data:
                        all_data[key] = []
                    
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        scores = expit(scores)
                        auc = roc_auc_score(labels, scores)
                        all_data[key].append(auc)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
    
    if not all_data:
        print("No valid data points found for experiment 3")
        return
    
    # Get unique values for plotting
    true_samples_list = sorted(list(set([k[1] for k in all_data.keys()])))
    upsampling_factors = sorted(list(set([k[2] for k in all_data.keys()])))
    
    # Create the multi-layer plot (1x at bottom, 5x at top)
    fig, axes = plt.subplots(len(upsampling_factors), 1, figsize=(12, 3*len(upsampling_factors)), 
                            sharex=True, sharey=False)
    if len(upsampling_factors) == 1:
        axes = [axes]
    
    colors = [f"C{i}" for i in range(len(architectures))]
    
    # Collect all AUC values for setting y-axis limits
    all_aucs = []
    
    # Plot each upsampling factor as a separate subplot
    for i, factor in enumerate(upsampling_factors):
        ax = axes[i]
        
        # Plot each architecture
        for arch_idx, architecture in enumerate(architectures):
            x_values = []
            y_values = []
            y_stds = []
            
            for samples in true_samples_list:
                key = (architecture, samples, factor)
                if key in all_data and all_data[key]:
                    x_values.append(samples)
                    if len(seeds) > 1:
                        y_values.append(np.mean(all_data[key]))
                        y_stds.append(np.std(all_data[key]))
                    else:
                        y_values.append(all_data[key][0])
                        y_stds.append(0)
            
            if x_values:
                color = colors[arch_idx]
                all_aucs.extend(y_values)
                if len(seeds) > 1 and any(y_stds):
                    ax.errorbar(x_values, y_values, yerr=y_stds, fmt='o-', 
                              label=architecture, color=color, linewidth=2, capsize=5, capthick=2)
                else:
                    ax.plot(x_values, y_values, 'o-', label=architecture, color=color, linewidth=2)
        
        # Set subtle label on the left side
        ax.text(-0.1, 0.5, f'{factor}x', transform=ax.transAxes, fontsize=14, 
                va='center', ha='center', rotation=90, alpha=0.7)
        ax.set_ylabel('AUC', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on the top subplot (5x)
        if i == len(upsampling_factors) - 1:
            ax.legend(fontsize=12)
        
        # Set y-axis limits based on data range for this subplot
        if all_aucs:
            min_auc = min(all_aucs)
            y_min = max(0.0, min_auc - 0.05)  # Start slightly below minimum
            y_max = 1.0
            ax.set_ylim(y_min, y_max)
        
        # Set larger font sizes for tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set x-axis properties for the bottom subplot (1x)
    axes[0].set_xlabel('Number of True Positive Samples', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].set_xticks(true_samples_list)
    axes[0].set_xticklabels(true_samples_list)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved experiment 3 line plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_experiment_2_recall_at_fpr(base_results_dir: Path, architectures: List[str], save_path=None, 
                                   fpr_target: float = 0.01, filtered: bool = True, seeds: List[str] = None):
    """Plot Recall at 1% FPR for all architectures in experiment 2."""
    from scipy.special import expit
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "2-spam-pred-auc-increasing-spam-fixed-total"
    
    plt.figure(figsize=(12, 8))
    colors = [f"C{i}" for i in range(len(architectures))]
    
    for arch_idx, architecture in enumerate(architectures):
        # Get all result files across seeds for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        
        if not seed_files:
            print(f"No result files found for experiment 2, architecture {architecture}")
            continue
        
        # Group by class1 count and get both train and eval results
        class1_to_train_files = {}
        class1_to_eval_files = {}
        
        for seed, files in seed_files.items():
            for file in files:
                match = re.search(r'class1_(\d+)', file)
                if match:
                    class1_count = int(match.group(1))
                    
                    # Check if this is an eval file (contains eval_on_ in filename)
                    if 'eval_on_' in file:
                        if class1_count not in class1_to_eval_files:
                            class1_to_eval_files[class1_count] = []
                        class1_to_eval_files[class1_count].append(file)
                    else:
                        if class1_count not in class1_to_train_files:
                            class1_to_train_files[class1_count] = []
                        class1_to_train_files[class1_count].append(file)
        
        color = colors[arch_idx]
        
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
        
        # Plot training results (solid lines)
        train_x = []
        train_recalls = []
        train_stds = []
        
        for class1_count in sorted(class1_to_train_files.keys()):
            files = class1_to_train_files[class1_count]
            recalls = []
            
            for file in files:
                try:
                    scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                    recall = recall_at_fpr_func(scores, labels)
                    recalls.append(recall)
                except Exception as e:
                    print(f"Error processing train file {file}: {e}")
                    continue
            
            if recalls:
                train_x.append(class1_count)
                train_recalls.append(np.mean(recalls))
                train_stds.append(np.std(recalls))
        
        # Plot training results (solid lines)
        if train_x:
            if len(seeds) > 1:
                plt.errorbar(train_x, train_recalls, yerr=train_stds, fmt='o-', 
                            label=f'{architecture}', linewidth=2, capsize=5, capthick=2, color=color)
            else:
                plt.plot(train_x, train_recalls, 'o-', label=f'{architecture}', linewidth=2, color=color)
        
        # Plot evaluation results (dashed lines) - if any exist
        eval_x = []
        eval_recalls = []
        eval_stds = []
        
        for class1_count in sorted(class1_to_eval_files.keys()):
            files = class1_to_eval_files[class1_count]
            recalls = []
            
            for file in files:
                try:
                    scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                    recall = recall_at_fpr_func(scores, labels)
                    recalls.append(recall)
                except Exception as e:
                    print(f"Error processing eval file {file}: {e}")
                    continue
            
            if recalls:
                eval_x.append(class1_count)
                eval_recalls.append(np.mean(recalls))
                eval_stds.append(np.std(recalls))
        
        # Plot evaluation results (dashed lines in same color) - only if eval data exists
        if eval_x:
            if len(seeds) > 1:
                plt.errorbar(eval_x, eval_recalls, yerr=eval_stds, fmt='o--', 
                            label=f'{architecture} Eval', linewidth=2, capsize=5, capthick=2, color=color)
            else:
                plt.plot(eval_x, eval_recalls, 'o--', label=f'{architecture} Eval', linewidth=2, color=color)
    
    plt.title(f"Experiment 2: Recall at {fpr_target*100}% FPR - All Architecture Probe Performance" + 
              (" (filtered)" if filtered else " (all)") + 
              (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""))
    plt.ylabel(f"Recall at {fpr_target*100}% FPR")
    plt.xlabel("Number of class 1 (positive) samples in train set")
    plt.xscale('log')
    
    # Set y-axis to start at a reasonable nonzero value based on the data
    all_recalls = []
    for arch_idx, architecture in enumerate(architectures):
        # Collect all recall values for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        if seed_files:
            for seed, files in seed_files.items():
                for file in files:
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        recall = recall_at_fpr_func(scores, labels)
                        all_recalls.append(recall)
                    except Exception:
                        continue
    
    if all_recalls:
        min_recall = min(all_recalls)
        y_min = max(0.0, min_recall - 0.1)  # Start at least 0.1 below the minimum, but not below 0
        plt.ylim(y_min, 1)
    else:
        plt.ylim(0, 1)  # Fallback range
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved experiment 2 recall@FPR plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_experiment_3_upsampling_lineplot_per_architecture(base_results_dir: Path, architectures: List[str], save_path=None, 
                                                          metric='auc', fpr_target: float = 0.01, filtered: bool = True, seeds: List[str] = None):
    """Plot experiment 3 as separate plots per architecture, each showing upsampling factors vs true samples."""
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "3-spam-pred-auc-llm-upsampling"
    
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
    
    # Collect data for all architectures
    all_data = {}
    
    for architecture in architectures:
        # Get all result files across seeds for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        
        if not seed_files:
            print(f"No result files found for experiment 3, architecture {architecture}")
            continue
        
        # Parse filenames to extract true samples and upsampling ratio
        for seed, files in seed_files.items():
            for file in files:
                # Extract true samples (n_real_pos) and upsampling ratio from rebuild suffix
                llm_match = re.search(r'llm_neg(\d+)_pos(\d+)_(\d+)x', file)
                
                if llm_match:
                    n_real_neg = int(llm_match.group(1))
                    n_real_pos = int(llm_match.group(2))  # This is the true number of positive samples
                    upsampling_factor = int(llm_match.group(3))
                    
                    key = (architecture, n_real_pos, upsampling_factor)
                    if key not in all_data:
                        all_data[key] = []
                    
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        if metric == 'auc':
                            scores = expit(scores)
                            value = roc_auc_score(labels, scores)
                        else:  # recall at fpr
                            value = recall_at_fpr_func(scores, labels)
                        all_data[key].append(value)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
    
    if not all_data:
        print("No valid data points found for experiment 3")
        return
    
    # Get unique values for plotting
    true_samples_list = sorted(list(set([k[1] for k in all_data.keys()])))
    upsampling_factors = sorted(list(set([k[2] for k in all_data.keys()])))
    
    # Define markers for different upsampling factors
    markers = ['o', 's', '^', 'D', '*']  # circle, square, triangle, diamond, star
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']  # Different colors for each upsampling factor
    
    # Create one plot per architecture
    for architecture in architectures:
        plt.figure(figsize=(10, 6))
        
        # Plot each upsampling factor as a separate line
        for i, factor in enumerate(upsampling_factors):
            x_values = []
            y_values = []
            y_stds = []
            
            for samples in true_samples_list:
                key = (architecture, samples, factor)
                if key in all_data and all_data[key]:
                    x_values.append(samples)
                    if len(seeds) > 1:
                        y_values.append(np.mean(all_data[key]))
                        y_stds.append(np.std(all_data[key]))
                    else:
                        y_values.append(all_data[key][0])
                        y_stds.append(0)
            
            if x_values:
                color = colors[i]
                marker = markers[i]
                if len(seeds) > 1 and any(y_stds):
                    plt.errorbar(x_values, y_values, yerr=y_stds, fmt=f'{marker}-', 
                              label=f'{factor}x', color=color, linewidth=2, capsize=5, capthick=2, markersize=8)
                else:
                    plt.plot(x_values, y_values, f'{marker}-', label=f'{factor}x', color=color, linewidth=2, markersize=8)
        
        # Set plot properties
        plt.xlabel('Number of True Positive Samples', fontsize=14)
        if metric == 'auc':
            plt.ylabel('AUC', fontsize=14)
            title = f"Experiment 3: {architecture.capitalize()} Probe Performance vs Upsampling"
        else:
            plt.ylabel(f'Recall at {fpr_target*100}% FPR', fontsize=14)
            title = f"Experiment 3: {architecture.capitalize()} Probe Performance vs Upsampling (Recall at {fpr_target*100}% FPR)"
        
        plt.title(title + (" (filtered)" if filtered else " (all)") + 
                 (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""), fontsize=16)
        
        plt.xscale('log')
        plt.xticks(true_samples_list, true_samples_list)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Set larger font sizes for tick labels
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Set y-axis limits for better readability
        all_values = []
        for factor in upsampling_factors:
            for samples in true_samples_list:
                key = (architecture, samples, factor)
                if key in all_data and all_data[key]:
                    all_values.extend(all_data[key])
        
        if all_values:
            min_val = min(all_values)
            y_min = max(0.0, min_val - 0.05)
            y_max = 1.0
            plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save individual plot for this architecture
        if save_path:
            # Create filename with architecture name
            base_path = Path(save_path)
            arch_save_path = base_path.parent / f"{base_path.stem}_{architecture}{base_path.suffix}"
            plt.savefig(arch_save_path, dpi=150, bbox_inches='tight')
            print(f"Saved experiment 3 {metric} plot for {architecture} to {arch_save_path}")
            plt.close()
        else:
            plt.show()


def plot_experiment_3_upsampling_lineplot_grid(base_results_dir: Path, architectures: List[str], save_path=None, 
                                              metric='auc', fpr_target: float = 0.01, filtered: bool = True, seeds: List[str] = None):
    """Plot experiment 3 as a grid of subplots, one per architecture, each showing upsampling factors vs true samples."""
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "3-spam-pred-auc-llm-upsampling"
    
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
    
    # Collect data for all architectures
    all_data = {}
    
    for architecture in architectures:
        # Get all result files across seeds for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        
        if not seed_files:
            print(f"No result files found for experiment 3, architecture {architecture}")
            continue
        
        # Parse filenames to extract true samples and upsampling ratio
        for seed, files in seed_files.items():
            for file in files:
                # Extract true samples (n_real_pos) and upsampling ratio from rebuild suffix
                llm_match = re.search(r'llm_neg(\d+)_pos(\d+)_(\d+)x', file)
                
                if llm_match:
                    n_real_neg = int(llm_match.group(1))
                    n_real_pos = int(llm_match.group(2))  # This is the true number of positive samples
                    upsampling_factor = int(llm_match.group(3))
                    
                    key = (architecture, n_real_pos, upsampling_factor)
                    if key not in all_data:
                        all_data[key] = []
                    
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        if metric == 'auc':
                            scores = expit(scores)
                            value = roc_auc_score(labels, scores)
                        else:  # recall at fpr
                            value = recall_at_fpr_func(scores, labels)
                        all_data[key].append(value)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
    
    if not all_data:
        print("No valid data points found for experiment 3")
        return
    
    # Get unique values for plotting
    true_samples_list = sorted(list(set([k[1] for k in all_data.keys()])))
    upsampling_factors = sorted(list(set([k[2] for k in all_data.keys()])))
    
    # Define markers for different upsampling factors
    markers = ['o', 's', '^', 'D', '*']  # circle, square, triangle, diamond, star
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']  # Different colors for each upsampling factor
    
    # Create grid layout
    n_architectures = len(architectures)
    n_cols = 2  # 2 columns
    n_rows = (n_architectures + 1) // 2  # Calculate rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each architecture
    for arch_idx, architecture in enumerate(architectures):
        ax = axes_flat[arch_idx]
        
        # Plot each upsampling factor as a separate line
        for i, factor in enumerate(upsampling_factors):
            x_values = []
            y_values = []
            y_stds = []
            
            for samples in true_samples_list:
                key = (architecture, samples, factor)
                if key in all_data and all_data[key]:
                    x_values.append(samples)
                    if len(seeds) > 1:
                        y_values.append(np.mean(all_data[key]))
                        y_stds.append(np.std(all_data[key]))
                    else:
                        y_values.append(all_data[key][0])
                        y_stds.append(0)
            
            if x_values:
                color = colors[i]
                marker = markers[i]
                if len(seeds) > 1 and any(y_stds):
                    ax.errorbar(x_values, y_values, yerr=y_stds, fmt=f'{marker}-', 
                              label=f'{factor}x', color=color, linewidth=2, capsize=5, capthick=2, markersize=8)
                else:
                    ax.plot(x_values, y_values, f'{marker}-', label=f'{factor}x', color=color, linewidth=2, markersize=8)
        
        # Set subplot properties
        ax.set_xlabel('Number of True Positive Samples', fontsize=12)
        if metric == 'auc':
            ax.set_ylabel('AUC', fontsize=12)
        else:
            ax.set_ylabel(f'Recall at {fpr_target*100}% FPR', fontsize=12)
        
        ax.set_title(f'{architecture.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(true_samples_list)
        ax.set_xticklabels(true_samples_list)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set larger font sizes for tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Set y-axis limits for better readability
        all_values = []
        for factor in upsampling_factors:
            for samples in true_samples_list:
                key = (architecture, samples, factor)
                if key in all_data and all_data[key]:
                    all_values.extend(all_data[key])
        
        if all_values:
            min_val = min(all_values)
            y_min = max(0.0, min_val - 0.05)
            y_max = 1.0
            ax.set_ylim(y_min, y_max)
    
    # Hide unused subplots
    for i in range(n_architectures, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add overall title
    if metric == 'auc':
        title = "Experiment 3: All Architecture Probe Performance vs Upsampling (AUC)"
    else:
        title = f"Experiment 3: All Architecture Probe Performance vs Upsampling (Recall at {fpr_target*100}% FPR)"
    
    plt.suptitle(title + (" (filtered)" if filtered else " (all)") + 
                 (f" (seeds: {', '.join(seeds)})" if len(seeds) > 1 else ""), 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved experiment 3 {metric} grid plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_experiment_3_upsampling_lineplot_recall(base_results_dir: Path, architectures: List[str], save_path=None, 
                                                fpr_target: float = 0.01, filtered: bool = True, seeds: List[str] = None):
    """Plot experiment 3 as a multi-layer line plot showing Recall at 1% FPR vs upsampling factors for all architectures."""
    from scipy.special import expit
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "3-spam-pred-auc-llm-upsampling"
    
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
    
    # Collect data for all architectures
    all_data = {}
    
    for architecture in architectures:
        # Get all result files across seeds for this architecture
        seed_files = _get_result_files_for_seeds(base_results_dir, seeds, experiment_folder, 
                                                architecture, 'dataclass_exps_94_better_spam')
        
        if not seed_files:
            print(f"No result files found for experiment 3, architecture {architecture}")
            continue
        
        # Parse filenames to extract true samples and upsampling ratio
        for seed, files in seed_files.items():
            for file in files:
                # Extract true samples (n_real_pos) and upsampling ratio from rebuild suffix
                llm_match = re.search(r'llm_neg(\d+)_pos(\d+)_(\d+)x', file)
                
                if llm_match:
                    n_real_neg = int(llm_match.group(1))
                    n_real_pos = int(llm_match.group(2))  # This is the true number of positive samples
                    upsampling_factor = int(llm_match.group(3))
                    
                    key = (architecture, n_real_pos, upsampling_factor)
                    if key not in all_data:
                        all_data[key] = []
                    
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        recall = recall_at_fpr_func(scores, labels)
                        all_data[key].append(recall)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
    
    if not all_data:
        print("No valid data points found for experiment 3")
        return
    
    # Get unique values for plotting
    true_samples_list = sorted(list(set([k[1] for k in all_data.keys()])))
    upsampling_factors = sorted(list(set([k[2] for k in all_data.keys()])))
    
    # Create the multi-layer plot (1x at bottom, 5x at top)
    fig, axes = plt.subplots(len(upsampling_factors), 1, figsize=(12, 3*len(upsampling_factors)), 
                            sharex=True, sharey=False)
    if len(upsampling_factors) == 1:
        axes = [axes]
    
    colors = [f"C{i}" for i in range(len(architectures))]
    
    # Collect all recall values for setting y-axis limits
    all_recalls = []
    
    # Plot each upsampling factor as a separate subplot
    for i, factor in enumerate(upsampling_factors):
        ax = axes[i]
        
        # Plot each architecture
        for arch_idx, architecture in enumerate(architectures):
            x_values = []
            y_values = []
            y_stds = []
            
            for samples in true_samples_list:
                key = (architecture, samples, factor)
                if key in all_data and all_data[key]:
                    x_values.append(samples)
                    if len(seeds) > 1:
                        y_values.append(np.mean(all_data[key]))
                        y_stds.append(np.std(all_data[key]))
                    else:
                        y_values.append(all_data[key][0])
                        y_stds.append(0)
            
            if x_values:
                color = colors[arch_idx]
                all_recalls.extend(y_values)
                if len(seeds) > 1 and any(y_stds):
                    ax.errorbar(x_values, y_values, yerr=y_stds, fmt='o-', 
                              label=architecture, color=color, linewidth=2, capsize=5, capthick=2)
                else:
                    ax.plot(x_values, y_values, 'o-', label=architecture, color=color, linewidth=2)
        
        # Set subtle label on the left side
        ax.text(-0.1, 0.5, f'{factor}x', transform=ax.transAxes, fontsize=14, 
                va='center', ha='center', rotation=90, alpha=0.7)
        ax.set_ylabel(f'Recall at {fpr_target*100}% FPR', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on the top subplot (5x)
        if i == len(upsampling_factors) - 1:
            ax.legend(fontsize=12)
        
        # Set y-axis limits based on data range for this subplot
        if all_recalls:
            min_recall = min(all_recalls)
            y_min = max(0.0, min_recall - 0.05)  # Start slightly below minimum
            y_max = 1.0
            ax.set_ylim(y_min, y_max)
        
        # Set larger font sizes for tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set x-axis properties for the bottom subplot (1x)
    axes[0].set_xlabel('Number of True Positive Samples', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].set_xticks(true_samples_list)
    axes[0].set_xticklabels(true_samples_list)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved experiment 3 recall@FPR line plot to {save_path}")
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
