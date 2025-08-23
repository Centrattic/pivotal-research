"""
Hyperparameter Sweep Visualization

This module creates hyperparameter sweep plots showing the effect of different
hyperparameters on performance across sample counts. Focuses on:
- Linear probes: C parameter  
- Attention probes: weight_decay parameter at each learning rate
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import seaborn as sns
from scipy import stats
import re

try:
    from .data_loader import (
        get_data_for_visualization,
        get_eval_datasets,
        get_run_names
    )
    from .viz_util import (
        setup_plot_style,
        format_run_name_for_display,
        format_dataset_name_for_display
    )
except ImportError:
    from data_loader import (
        get_data_for_visualization,
        get_eval_datasets,
        get_run_names
    )
    from viz_util import (
        setup_plot_style,
        format_run_name_for_display,
        format_dataset_name_for_display
    )


def get_hyperparameter_sweep_data(
    eval_dataset: str,
    experiment: str,
    run_name: str,
    probe_type: str,
    metric: str = 'auc'
) -> pd.DataFrame:
    """
    Get data for hyperparameter sweep analysis.
    
    Args:
        eval_dataset: Evaluation dataset name
        experiment: Experiment type ('2-' or '4-')
        run_name: Model run name
        probe_type: Type of probe ('sklearn_linear', 'attention')
        metric: Metric to plot ('auc' or 'recall')
        
    Returns:
        DataFrame with hyperparameter sweep data
    """
    # Get data for the experiment
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        exclude_attention=False,
        include_val_eval=False  # Only use test_eval
    )
    
    if df.empty:
        return df
    
    # Filter by probe type
    if probe_type == 'sklearn_linear':
        df = df[df['probe_name'].str.contains('sklearn_linear', na=False)]
    elif probe_type == 'attention':
        df = df[df['probe_name'].str.contains('attention', na=False)]
    
    return df


def plot_linear_c_sweep(
    eval_dataset: str,
    experiment: str,
    run_name: str,
    save_path: Path,
    metric: str = 'auc'
):
    """
    Plot linear C hyperparameter sweep.
    
    Args:
        eval_dataset: Evaluation dataset name
        experiment: Experiment type ('2-' or '4-')
        run_name: Model run name
        save_path: Path to save the plot
        metric: Metric to plot ('auc' or 'recall')
    """
    # Get data
    df = get_hyperparameter_sweep_data(eval_dataset, experiment, run_name, 'sklearn_linear', metric)
    
    if df.empty:
        print(f"No linear probe data found for {eval_dataset}, {run_name}")
        return
    
    # Get unique C values and sample counts, excluding default C=1.0 and extreme values with partial results
    c_values = sorted([c for c in df['C'].dropna().unique() 
                      if c != 1.0 and c not in [1e-05, 1e-04, 1e+04, 1e+05]])
    sample_counts = sorted(df['num_positive_samples'].dropna().unique())
    
    if not c_values or not sample_counts:
        print(f"Insufficient data for linear C sweep: {eval_dataset}, {run_name}")
        return
    
    # Create performance matrix
    performance_matrix = np.zeros((len(sample_counts), len(c_values)))
    probe_counts_matrix = np.zeros((len(sample_counts), len(c_values)), dtype=int)
    
    for i, sample_count in enumerate(sample_counts):
        for j, c_val in enumerate(c_values):
            mask = (df['num_positive_samples'] == sample_count) & (df['C'] == c_val)
            subset = df[mask]
            if not subset.empty:
                # Average over seeds for this specific C value and sample count
                performance_matrix[i, j] = subset[metric].mean()
                probe_counts_matrix[i, j] = len(subset)
            else:
                performance_matrix[i, j] = np.nan
                probe_counts_matrix[i, j] = 0
    
    # Create plot with smaller size
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with viridis colormap
    im = ax.imshow(performance_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_xticks(range(len(c_values)))
    ax.set_xticklabels([f'{c_val:.1e}' if c_val != 1.0 else '1.0' for c_val in c_values])
    ax.set_yticks(range(len(sample_counts)))
    ax.set_yticklabels([f'{int(sc)}' for sc in sample_counts])
    
    ax.set_xlabel('Linear C parameter')
    ax.set_ylabel('Number of positive samples')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric.upper()}', rotation=270, labelpad=20)
    
    # Add title with bigger font
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    experiment_name = "Unbalanced" if experiment == '2-' else "Balanced"
    ax.set_title(f'{formatted_run_name} Linear Probes - {experiment_name} {formatted_dataset} {metric.upper()} vs C', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_weight_decay_sweep(
    eval_dataset: str,
    experiment: str,
    run_name: str,
    save_path: Path,
    metric: str = 'auc'
):
    """
    Plot attention weight_decay hyperparameter sweep for each learning rate.
    
    Args:
        eval_dataset: Evaluation dataset name
        experiment: Experiment type ('2-' or '4-')
        run_name: Model run name
        save_path: Path to save the plot
        metric: Metric to plot ('auc' or 'recall')
    """
    # Get data
    df = get_hyperparameter_sweep_data(eval_dataset, experiment, run_name, 'attention', metric)
    
    if df.empty:
        print(f"No attention probe data found for {eval_dataset}, {run_name}")
        return
    
    # Get unique learning rates, excluding default lr=5e-4
    all_lr_values = sorted(df['lr'].dropna().unique())
    lr_values = sorted([lr for lr in all_lr_values if abs(lr - 5e-4) > 1e-6])
    
    print(f"Available learning rates for {eval_dataset}, {run_name}: {all_lr_values}")
    print(f"After excluding default (5e-4): {lr_values}")
    
    if not lr_values:
        print(f"No learning rate data found for attention probes: {eval_dataset}, {run_name}")
        return
    
    # Create subplot for each learning rate with smaller figure size
    n_lrs = len(lr_values)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, lr in enumerate(lr_values):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Filter data for this learning rate
        lr_df = df[df['lr'] == lr]
        
        if lr_df.empty:
            continue
        
        # Get unique weight_decay values and sample counts, excluding default weight_decay=0.0
        all_wd_values = sorted(lr_df['weight_decay'].dropna().unique())
        wd_values = sorted([wd for wd in all_wd_values if wd != 0.0])
        sample_counts = sorted(lr_df['num_positive_samples'].dropna().unique())
        
        print(f"  LR {lr:.1e}: weight_decay values {all_wd_values} -> {wd_values}")
        print(f"  LR {lr:.1e}: sample counts {sample_counts}")
        
        if not wd_values or not sample_counts:
            continue
        
        # Create performance matrix
        performance_matrix = np.zeros((len(sample_counts), len(wd_values)))
        
        for i, sample_count in enumerate(sample_counts):
            for j, wd in enumerate(wd_values):
                mask = (lr_df['num_positive_samples'] == sample_count) & (lr_df['weight_decay'] == wd)
                subset = lr_df[mask]
                if not subset.empty:
                    # Average over seeds for this specific weight decay and sample count
                    performance_matrix[i, j] = subset[metric].mean()
                else:
                    performance_matrix[i, j] = np.nan
        
        # Create heatmap
        im = ax.imshow(performance_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Set labels
        ax.set_xticks(range(len(wd_values)))
        ax.set_xticklabels([f'{wd:.1e}' for wd in wd_values], rotation=45)
        ax.set_yticks(range(len(sample_counts)))
        ax.set_yticklabels([f'{int(sc)}' for sc in sample_counts])
        
        ax.set_xlabel('Weight Decay')
        ax.set_ylabel('Num Positive Samples')
        ax.set_title(f'LR = {lr:.1e}', fontsize=12)
    
    # Hide unused subplots
    for idx in range(n_lrs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title with bigger font and single line
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    experiment_name = "Unbalanced" if experiment == '2-' else "Balanced"
    fig.suptitle(f'{formatted_run_name} Attention Probes - {experiment_name} {formatted_dataset} {metric.upper()} vs Weight Decay', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_hyperparameter_sweeps(skip_existing: bool = False):
    """
    Generate all hyperparameter sweep plots.
    
    Args:
        skip_existing: If True, skip generating plots that already exist
    """
    # Create output directory
    output_dir = Path("visualizations/hyp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Focus only on specific datasets and models
    target_datasets = ['94_better_spam', '98_mask_all_honesty']
    target_models = ['spam_gemma_9b', 'mask_llama33_70b']
    
    # Focus only on experiments 2- (imbalanced) and 4- (balanced)
    experiments = ['2-', '4-']
    
    # Generate hyperparameter sweep plots
    for run_name in target_models:
        print(f"\nProcessing hyperparameter sweeps for: {run_name}")
        
        for eval_dataset in target_datasets:
            for experiment in experiments:
                print(f"  Processing {experiment} for {eval_dataset}")
                
                for metric in ['auc', 'recall']:
                    # Linear C sweep
                    save_path = output_dir / f"linear_c_sweep_{metric}_{eval_dataset}_{run_name}_{experiment}.png"
                    if not skip_existing or not save_path.exists():
                        plot_linear_c_sweep(eval_dataset, experiment, run_name, save_path, metric)
                    
                    # Attention weight_decay sweep
                    save_path = output_dir / f"attention_wd_sweep_{metric}_{eval_dataset}_{run_name}_{experiment}.png"
                    if not skip_existing or not save_path.exists():
                        plot_attention_weight_decay_sweep(eval_dataset, experiment, run_name, save_path, metric)
    
    print(f"Hyperparameter sweep generation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    generate_all_hyperparameter_sweeps()
