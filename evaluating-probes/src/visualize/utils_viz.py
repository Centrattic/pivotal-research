
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
    plt.figure(figsize=(6, 4))  # Larger figure size to accommodate longer titles
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
    plt.xlabel("Logit difference", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Logit difference histogram from CSV", fontsize=14)
    plt.yscale("log")
    plt.legend(fontsize=10)
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
    
    plt.figure(figsize=(6, 4))  # Larger figure size to accommodate longer titles
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
    
    plt.title("Varying number of positive training examples\nwith 3000 negative examples", fontsize=14)
    plt.ylabel("Recall", fontsize=12)
    plt.xlabel("Number of positive examples in the train set", fontsize=12)
    plt.xscale('log')
    
    # Set y-axis to start at a reasonable nonzero value based on the data
    if all_recalls:
        min_recall = min(all_recalls)
        y_min = max(0.0, min_recall - 0.1)  # Start at least 0.1 below the minimum, but not below 0
        plt.ylim(y_min, 1)
    else:
        plt.ylim(0, 1)
    
    plt.legend(fontsize=10)
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
    
    plt.figure(figsize=(6, 4))  # Larger figure size to accommodate longer titles
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
    
    plt.title("Varying number of positive training examples\nwith 3000 negative examples", fontsize=14)
    plt.ylabel("AUC", fontsize=12)
    plt.xlabel("Number of positive examples in the train set", fontsize=12)
    plt.xscale('log')
    
    # Set y-axis to start at a reasonable nonzero value based on the data
    if all_aucs:
        min_auc = min(all_aucs)
        y_min = max(0.0, min_auc - 0.1)  # Start at least 0.1 below the minimum, but not below 0
        plt.ylim(y_min, 1)
    else:
        plt.ylim(0, 1)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved multi-folder AUC vs #class1 plot to {save_path}")
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
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows), squeeze=False)  # Smaller figure size for bigger text
    
    for idx, log_path in enumerate(log_files[:max_probes]):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        with open(log_path, 'r') as f:
            d = json.load(f)
        loss_history = d.get('loss_history', [])
        ax.plot(loss_history)
        label = extract_probe_info_from_filename(log_path)
        ax.set_title(label, fontsize=10)  # Bigger font size
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
    
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


# Removed: plot_experiment_2_per_seed - replaced by plot_experiment_2_unified


def plot_experiment_3_per_probe(base_results_dir: Path, save_path_base=None, 
                                filtered: bool = True, seeds: List[str] = None, fpr_target: float = 0.01,
                                train_dataset: str = None):
    """Plot experiment 3 with separate plots for each individual probe, showing upsampling factors as different lines for both AUC and Recall at FPR."""
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "3-spam-pred-auc-llm-upsampling"
    
    # Collect data for all probes and seeds
    all_data_auc = {}
    all_data_recall = {}
    probe_names = set()
    eval_datasets = set()
    
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
    
    # Get all result files across all seeds
    for seed in seeds:
        seed_dir = base_results_dir / f"seed_{seed}" / experiment_folder / 'dataclass_exps_94_better_spam'
        if not seed_dir.exists():
            continue
        
        # Find all result files
        result_files = list(seed_dir.glob("*_results.json"))
        
        for file in result_files:
            # Extract probe name, true samples and upsampling ratio from filename
            # Example filename: eval_on_94_better_spam__train_on_94_better_spam_linear_max_L20_resid_post_llm_neg3000_pos10_4x_seed42_max_results.json
            llm_match = re.search(r'llm_neg(\d+)_pos(\d+)_(\d+)x', str(file))
            probe_match = re.search(r'train_on_94_better_spam_(.*?)_L\d+_resid_post_llm', str(file))
            
            # Extract evaluation dataset from filename
            eval_match = re.search(r'eval_on_([^_]+(?:_[^_]+)*?)__', str(file))
            
            if llm_match and probe_match and eval_match:
                n_real_pos = int(llm_match.group(2))
                upsampling_factor = int(llm_match.group(3))
                probe_name = probe_match.group(1)
                eval_dataset = eval_match.group(1)
                
                probe_names.add(probe_name)
                eval_datasets.add(eval_dataset)
                key = (probe_name, n_real_pos, upsampling_factor)
                
                # Initialize data structures
                if key not in all_data_auc:
                    all_data_auc[key] = []
                if key not in all_data_recall:
                    all_data_recall[key] = []
                
                try:
                    scores, labels = _get_scores_and_labels_from_result_file(str(file), filtered=filtered)
                    
                    # Calculate AUC
                    scores_auc = expit(scores)
                    auc = roc_auc_score(labels, scores_auc)
                    all_data_auc[key].append(auc)
                    
                    # Calculate Recall at FPR
                    recall = recall_at_fpr_func(scores, labels)
                    all_data_recall[key].append(recall)
                    
                except Exception as e:
                    continue
    
    if not all_data_auc:
        print("No valid data points found for experiment 3")
        return
    
    # Get unique values for plotting
    true_samples_list = sorted(list(set([k[1] for k in all_data_auc.keys()])))
    upsampling_factors = sorted(list(set([k[2] for k in all_data_auc.keys()])))
    
    # Filter out 10x upsampling for now
    upsampling_factors = [f for f in upsampling_factors if f != 10]
    
    # Determine if any evaluation is out-of-distribution
    is_ood = False
    if train_dataset is not None:
        for eval_dataset in eval_datasets:
            if eval_dataset != train_dataset:
                is_ood = True
                break
    
    # Create gradient red colors for upsampling factors (light red to dark red)
    red_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(upsampling_factors)))
    
    # Define markers for different upsampling factors
    markers = ['o', 's', '^', 'D', '*']  # circle, square, triangle, diamond, star
    
    # Create separate plots for each probe (AUC and Recall)
    for probe_name in sorted(probe_names):
        # Create AUC plot
        plt.figure(figsize=(6, 4))  # Larger figure size to accommodate longer titles
        
        # Plot each upsampling factor as a separate line
        for i, factor in enumerate(upsampling_factors):
            x_values = []
            medians = []
            lower_bounds = []
            upper_bounds = []
            
            for samples in true_samples_list:
                key = (probe_name, samples, factor)
                if key in all_data_auc and all_data_auc[key]:
                    x_values.append(samples)
                    
                    # Calculate median and percentiles for error bands
                    values = all_data_auc[key]
                    medians.append(np.median(values))
                    
                    # Calculate 25th and 75th percentiles for shaded region
                    if len(values) > 1:
                        # Use numpy percentile for more accurate calculation
                        lower_bounds.append(np.percentile(values, 25))
                        upper_bounds.append(np.percentile(values, 75))
                    else:
                        # If only one value, use the same value for bounds (no shading)
                        lower_bounds.append(values[0])
                        upper_bounds.append(values[0])
            
            if x_values:
                color = red_colors[i]  # Use gradient red color based on upsampling factor
                marker = markers[i] if i < len(markers) else 'o'  # Use different marker for each upsampling factor
                
                # Plot median line
                plt.plot(x_values, medians, f'{marker}-', label=f'{factor}x', color=color, linewidth=2, markersize=8)
                
                # Add shaded confidence band (25th to 75th percentile)
                if len(x_values) > 0 and lower_bounds and upper_bounds:
                    # Only show shading if there's actual variation (not all bounds are the same)
                    has_variation = any(lower_bounds[j] != upper_bounds[j] for j in range(len(lower_bounds)))
                    if has_variation:
                        plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color=color)
        
        # Set plot properties for AUC
        plt.xlabel('Number of positive real examples in the train set', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        if is_ood:
            plt.title("LLM Upsampling: Out of Distribution", fontsize=14)
        else:
            plt.title("LLM Upsampling", fontsize=14)
        
        plt.xticks(true_samples_list, true_samples_list)
        plt.grid(True, alpha=0.3)
        plt.legend(title="LLM Upsampling Factor", fontsize=10)
        
        # Set larger font sizes for tick labels
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # Set y-axis limits for better readability
        all_values = []
        for factor in upsampling_factors:
            for samples in true_samples_list:
                key = (probe_name, samples, factor)
                if key in all_data_auc and all_data_auc[key]:
                    all_values.extend(all_data_auc[key])
        
        if all_values:
            min_val = min(all_values)
            y_min = max(0.0, min_val - 0.05)
            y_max = 1.0
            plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save AUC plot
        if save_path_base:
            # Create filename with probe name and metric
            base_path = Path(save_path_base)
            probe_save_path = base_path.parent / f"{base_path.stem}_{probe_name}_auc{base_path.suffix}"
            plt.savefig(probe_save_path, dpi=150, bbox_inches='tight')
            print(f"Saved experiment 3 AUC plot for {probe_name} to {probe_save_path}")
            plt.close()
        else:
            plt.show()
        
        # Create Recall at FPR plot
        plt.figure(figsize=(6, 4))  # Larger figure size to accommodate longer titles
        
        # Plot each upsampling factor as a separate line
        for i, factor in enumerate(upsampling_factors):
            x_values = []
            medians = []
            lower_bounds = []
            upper_bounds = []
            
            for samples in true_samples_list:
                key = (probe_name, samples, factor)
                if key in all_data_recall and all_data_recall[key]:
                    x_values.append(samples)
                    
                    # Calculate median and percentiles for error bands
                    values = all_data_recall[key]
                    medians.append(np.median(values))
                    
                    # Calculate 25th and 75th percentiles for shaded region
                    if len(values) > 1:
                        # Use numpy percentile for more accurate calculation
                        lower_bounds.append(np.percentile(values, 25))
                        upper_bounds.append(np.percentile(values, 75))
                    else:
                        # If only one value, use the same value for bounds (no shading)
                        lower_bounds.append(values[0])
                        upper_bounds.append(values[0])
            
            if x_values:
                color = red_colors[i]  # Use gradient red color based on upsampling factor
                marker = markers[i] if i < len(markers) else 'o'  # Use different marker for each upsampling factor
                
                # Plot median line
                plt.plot(x_values, medians, f'{marker}-', label=f'{factor}x', color=color, linewidth=2, markersize=8)
                
                # Add shaded confidence band (25th to 75th percentile)
                if len(x_values) > 0 and lower_bounds and upper_bounds:
                    # Only show shading if there's actual variation (not all bounds are the same)
                    has_variation = any(lower_bounds[j] != upper_bounds[j] for j in range(len(lower_bounds)))
                    if has_variation:
                        plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color=color)
        
        # Set plot properties for Recall
        plt.xlabel('Number of positive real examples in the train set', fontsize=12)
        plt.ylabel(f'Recall at {fpr_target*100}% FPR', fontsize=12)
        if is_ood:
            plt.title("LLM Upsampling: Out of Distribution", fontsize=14)
        else:
            plt.title("LLM Upsampling", fontsize=14)
        
        plt.xticks(true_samples_list, true_samples_list)
        plt.grid(True, alpha=0.3)
        plt.legend(title="LLM Upsampling\nFactor", fontsize=10)
        
        # Set larger font sizes for tick labels
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # Set y-axis limits for better readability
        all_values = []
        for factor in upsampling_factors:
            for samples in true_samples_list:
                key = (probe_name, samples, factor)
                if key in all_data_recall and all_data_recall[key]:
                    all_values.extend(all_data_recall[key])
        
        if all_values:
            min_val = min(all_values)
            y_min = max(0.0, min_val - 0.05)
            y_max = 1.0
            plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save Recall plot
        if save_path_base:
            # Create filename with probe name and metric
            base_path = Path(save_path_base)
            probe_save_path = base_path.parent / f"{base_path.stem}_{probe_name}_recall{base_path.suffix}"
            plt.savefig(probe_save_path, dpi=150, bbox_inches='tight')
            print(f"Saved experiment 3 Recall plot for {probe_name} to {probe_save_path}")
            plt.close()
        else:
            plt.show()


# Removed: plot_experiment_2_total_with_error_bars - replaced by plot_experiment_2_unified
# Removed: plot_experiment_2_recall_total_with_error_bars - replaced by plot_experiment_2_unified


def get_best_probes_by_type(base_results_dir: Path, seeds: List[str], filtered: bool = True, eval_dataset: str = None) -> Dict[str, str]:
    """
    Determine the best performing probe of each type based on median AUC across all class1 counts.
    
    Args:
        base_results_dir: Path to results directory
        seeds: List of seeds to use
        filtered: Whether to use filtered scores
        eval_dataset: Evaluation dataset name (e.g., '87_is_spam', '94_better_spam')
    
    Returns:
        Dict mapping probe type to best probe name:
        - 'linear': best linear probe (last, max, mean, softmax)
        - 'sae': best SAE probe (16k_l0_408, 262k_l0_259)
        - 'attention': attention probe
        - 'act_sim': best activation similarity probe (max_max, last_last)
    """
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    experiment_folder = "2-spam-pred-auc-increasing-spam-fixed-total"
    dataclass_folder = "dataclass_exps_94_better_spam"
    
    # Define probe type patterns
    probe_patterns = {
        'linear': ['linear_last', 'linear_max', 'linear_mean', 'linear_softmax'],
        'sae': ['sae_16k_l0_408', 'sae_262k_l0_259'],
        'attention': ['attention_attention'],
        'act_sim': ['act_sim_max_max', 'act_sim_last_last']
    }
    
    best_probes = {}
    
    for probe_type, patterns in probe_patterns.items():
        probe_scores = {}
        
        for pattern in patterns:
            all_aucs = []
            
            # Get all result files for this probe pattern across seeds
            for seed in seeds:
                seed_dir = base_results_dir / f"seed_{seed}" / experiment_folder / dataclass_folder
                if not seed_dir.exists():
                    continue
                
                # Find files matching this pattern and evaluation dataset
                if eval_dataset:
                    result_files = glob.glob(str(seed_dir / f"eval_on_{eval_dataset}__*{pattern}*_results.json"))
                else:
                    result_files = glob.glob(str(seed_dir / f"*{pattern}*_results.json"))
                
                for file in result_files:
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        scores = expit(scores)
                        auc = roc_auc_score(labels, scores)
                        all_aucs.append(auc)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        continue
            
            if all_aucs:
                median_auc = np.median(all_aucs)
                probe_scores[pattern] = median_auc
        
        # Select the best probe for this type
        if probe_scores:
            best_probe = max(probe_scores.items(), key=lambda x: x[1])[0]
            best_probes[probe_type] = best_probe
            print(f"Best {probe_type} probe: {best_probe} (median AUC: {probe_scores[best_probe]:.3f})")
    
    return best_probes


def plot_experiment_2_unified(base_results_dir: Path, probe_names: List[str], save_path=None, 
                             metric='auc', fpr_target: float = 0.01, filtered: bool = True, 
                             seeds: List[str] = None, plot_title: str = None, eval_dataset: str = None,
                             probe_labels: Dict[str, str] = None):
    """
    Unified experiment 2 plotting function that can handle any selection of probes.
    
    Args:
        base_results_dir: Path to results directory
        probe_names: List of probe names to plot (e.g., ['linear_last', 'sae_16k_l0_408', 'attention_attention'])
        save_path: Path to save the plot
        metric: 'auc' or 'recall_at_fpr'
        fpr_target: FPR target for recall calculation (default 0.01)
        filtered: Whether to use filtered scores
        seeds: List of seeds to use
        plot_title: Custom title for the plot
        eval_dataset: Evaluation dataset name (e.g., '87_is_spam', '94_better_spam')
        probe_labels: Dictionary mapping probe config names to readable labels
    """
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    
    if seeds is None:
        seeds = ['42']
    
    experiment_folder = "2-spam-pred-auc-increasing-spam-fixed-total"
    dataclass_folder = "dataclass_exps_94_better_spam"
    
    # Determine if this is out-of-distribution evaluation
    # Training dataset is 94_better_spam (from dataclass_folder name)
    train_dataset = "94_better_spam"
    is_ood = (eval_dataset is not None and eval_dataset != train_dataset)
    
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
    
    plt.figure(figsize=(6, 4))  # Larger figure size to accommodate longer titles
    colors = [f"C{i}" for i in range(len(probe_names))]
    
    for probe_idx, probe_name in enumerate(probe_names):
        # Collect data across all seeds
        all_data = {}
        
        for seed in seeds:
            seed_dir = base_results_dir / f"seed_{seed}" / experiment_folder / dataclass_folder
            if not seed_dir.exists():
                continue
            
            # Find files matching this probe name and evaluation dataset
            if eval_dataset:
                result_files = glob.glob(str(seed_dir / f"eval_on_{eval_dataset}__*{probe_name}*_results.json"))
            else:
                result_files = glob.glob(str(seed_dir / f"*{probe_name}*_results.json"))
            
            for file in result_files:
                # Extract class1 count from filename
                match = re.search(r'class1_(\d+)', file)
                if match:
                    class1_count = int(match.group(1))
                    if class1_count not in all_data:
                        all_data[class1_count] = []
                    
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        if metric == 'auc':
                            scores = expit(scores)
                            value = roc_auc_score(labels, scores)
                        else:  # recall at fpr
                            value = recall_at_fpr_func(scores, labels)
                        all_data[class1_count].append(value)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
        
        # Calculate medians and confidence intervals
        x_values = []
        medians = []
        lower_bounds = []
        upper_bounds = []
        
        for class1_count in sorted(all_data.keys()):
            if all_data[class1_count]:
                x_values.append(class1_count)
                medians.append(np.median(all_data[class1_count]))
                
                # Calculate confidence interval (25th and 75th percentiles for interquartile range)
                if len(all_data[class1_count]) > 1:
                    # Use numpy percentile for more accurate calculation
                    lower_bounds.append(np.percentile(all_data[class1_count], 25))
                    upper_bounds.append(np.percentile(all_data[class1_count], 75))
                else:
                    # If only one value, use the same value for bounds (no shading)
                    value = all_data[class1_count][0]
                    lower_bounds.append(value)
                    upper_bounds.append(value)
        
        if x_values:
            color = colors[probe_idx]
            # Use readable label if available, otherwise use original name
            label = probe_labels.get(probe_name, probe_name)
            # Plot median line
            plt.plot(x_values, medians, 'o-', label=label, linewidth=2, color=color, markersize=6)
            # Add shaded confidence band (25th to 75th percentile)
            if lower_bounds and upper_bounds:
                # Only show shading if there's actual variation (not all bounds are the same)
                has_variation = any(lower_bounds[j] != upper_bounds[j] for j in range(len(lower_bounds)))
                if has_variation:
                    plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color=color)
    
    # Set plot properties
    if plot_title:
        title = plot_title
    else:
        title = "Varying number of positive training examples\nwith 3000 negative examples"
    
    plt.title(title, fontsize=14)
    
    # Collect all values to determine y-axis limits
    all_lower_bounds = []  # Store the 25th percentile (lower bound of confidence bands)
    all_upper_bounds = []  # Store the 75th percentile (upper bound of confidence bands)
    
    for probe_idx, probe_name in enumerate(probe_names):
        for seed in seeds:
            seed_dir = base_results_dir / f"seed_{seed}" / experiment_folder / dataclass_folder
            if not seed_dir.exists():
                continue
            
            # Find files matching this probe name and evaluation dataset
            if eval_dataset:
                result_files = glob.glob(str(seed_dir / f"eval_on_{eval_dataset}__*{probe_name}*_results.json"))
            else:
                result_files = glob.glob(str(seed_dir / f"*{probe_name}*_results.json"))
            
            # Group by class1 count to calculate confidence bands
            class1_to_values = {}
            for file in result_files:
                match = re.search(r'class1_(\d+)', file)
                if match:
                    class1_count = int(match.group(1))
                    if class1_count not in class1_to_values:
                        class1_to_values[class1_count] = []
                    
                    try:
                        scores, labels = _get_scores_and_labels_from_result_file(file, filtered=filtered)
                        if metric == 'auc':
                            scores = expit(scores)
                            value = roc_auc_score(labels, scores)
                        else:  # recall at fpr
                            value = recall_at_fpr_func(scores, labels)
                        class1_to_values[class1_count].append(value)
                    except Exception as e:
                        continue
            
            # Calculate confidence bands for this probe
            for class1_count, values in class1_to_values.items():
                if values:
                    sorted_values = np.sort(values)
                    n = len(sorted_values)
                    q25_idx = max(0, int(0.25 * n))
                    q75_idx = min(n - 1, int(0.75 * n))
                    all_lower_bounds.append(sorted_values[q25_idx])
                    all_upper_bounds.append(sorted_values[q75_idx])
    
    # Set y-axis limits based on confidence bands
    if all_lower_bounds and all_upper_bounds:
        # Find the lowest point of any confidence band and set y-axis 0.05 below it
        lowest_confidence_bound = min(all_lower_bounds)
        y_min = max(0.0, lowest_confidence_bound - 0.05)  # Don't go below 0
        y_max = 1.0
        plt.ylim(y_min, y_max)
    else:
        # Fallback limits
        if metric == 'auc':
            plt.ylim(0.4, 1)
        else:
            plt.ylim(0, 1)
    
    if metric == 'auc':
        if is_ood:
            plt.ylabel("AUC on text spam", fontsize=12)
        else:
            plt.ylabel("AUC", fontsize=12)
    else:
        plt.ylabel(f"Recall at {fpr_target*100}% FPR", fontsize=12)
    
    if is_ood:
        plt.xlabel("Number of positive email spam samples", fontsize=12)
    else:
        plt.xlabel("Number of positive examples in the train set", fontsize=12)
    plt.xscale('log')  # Keep log scale for experiment 2
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Increase tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved experiment 2 plot to {save_path}")
        plt.close()
    else:
        plt.show()
