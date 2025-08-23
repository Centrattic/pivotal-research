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
    
    # Get unique C values and sample counts
    c_values = sorted(df['C'].dropna().unique())
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
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    
    # Add title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    experiment_name = "Unbalanced" if experiment == '2-' else "Balanced"
    ax.set_title(f'{formatted_run_name} Linear Probes - {experiment_name} {formatted_dataset}\n{metric.upper()} vs C parameter')
    
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
    
    # Get unique learning rates
    lr_values = sorted(df['lr'].dropna().unique())
    
    if not lr_values:
        print(f"No learning rate data found for attention probes: {eval_dataset}, {run_name}")
        return
    
    # Create subplot for each learning rate
    n_lrs = len(lr_values)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, lr in enumerate(lr_values):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Filter data for this learning rate
        lr_df = df[df['lr'] == lr]
        
        if lr_df.empty:
            continue
        
        # Get unique weight_decay values and sample counts
        wd_values = sorted(lr_df['weight_decay'].dropna().unique())
        sample_counts = sorted(lr_df['num_positive_samples'].dropna().unique())
        
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
        ax.set_title(f'LR = {lr:.1e}')
    
    # Hide unused subplots
    for idx in range(n_lrs, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    experiment_name = "Unbalanced" if experiment == '2-' else "Balanced"
    fig.suptitle(f'{formatted_run_name} Attention Probes - {experiment_name} {formatted_dataset}\n{metric.upper()} vs Weight Decay (by Learning Rate)', 
                 fontsize=16, y=0.98)
    
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
    
    # Get available configurations
    eval_datasets = get_eval_datasets()
    run_names = get_run_names()
    
    # Filter datasets - exclude gen_eval datasets like 87_is_spam
    def _leading_id(name: str) -> int:
        try:
            return int(str(name).split('_', 1)[0])
        except Exception:
            return -1
    
    eval_datasets = [d for d in eval_datasets if _leading_id(d) == -1 or _leading_id(d) < 99]
    
    # Filter run_names to exclude Qwen models
    non_qwen_models = [name for name in run_names if 'qwen' not in name.lower()]
    
    # Focus only on experiments 2- (imbalanced) and 4- (balanced)
    experiments = ['2-', '4-']
    
    # Generate hyperparameter sweep plots
    for run_name in non_qwen_models:
        print(f"\nProcessing hyperparameter sweeps for: {run_name}")
        
        for eval_dataset in eval_datasets:
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
