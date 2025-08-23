"""
Hyperparameter Sweep Visualization

This module creates dedicated hyperparameter sweep plots showing performance
on test set with error bars over seeds for specific configurations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import seaborn as sns
from scipy import stats
import re

from .data_loader import (
    get_data_for_visualization,
    get_eval_datasets,
    get_run_names,
    load_metrics_data,
    extract_info_from_filename
)
from .plot_generator import (
    setup_plot_style,
    get_probe_label,
    get_detailed_probe_label,
    calculate_confidence_interval,
    format_run_name_for_display,
    format_dataset_name_for_display
)


def get_hyperparameter_sweep_data(
    eval_dataset: str,
    experiment: str,
    run_name: str,
    probe_type: str,
    hyperparam_name: str,
    num_positive_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Get data for hyperparameter sweep analysis.
    
    Args:
        eval_dataset: Evaluation dataset name
        experiment: Experiment type (e.g., '2-', '4-')
        run_name: Model run name
        probe_type: Type of probe ('sae', 'sklearn_linear', 'attention')
        hyperparam_name: Name of hyperparameter to sweep ('C', 'topk', 'lr', 'weight_decay')
        num_positive_samples: Filter by number of positive samples
        
    Returns:
        DataFrame with hyperparameter sweep data
    """
    # Get data with val_eval included for potential future use
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        exclude_attention=False,
        include_val_eval=True
    )
    
    if df.empty:
        return df
    
    # Filter by probe type
    if probe_type == 'sae':
        df = df[df['probe_name'].str.contains('sae', na=False)]
    elif probe_type == 'sklearn_linear':
        df = df[df['probe_name'].str.contains('sklearn_linear', na=False)]
    elif probe_type == 'attention':
        df = df[df['probe_name'].str.contains('attention', na=False)]
    
    # Filter by number of positive samples if specified
    if num_positive_samples is not None:
        df = df[df['num_positive_samples'] == num_positive_samples]
    
    # Filter to only include test_eval results
    df = df[df['filename'].str.contains('/test_eval/')]
    
    return df


def plot_linear_probe_c_sweep(
    eval_dataset: str,
    run_name: str,
    save_path: Path,
    experiment: str = '2-',
    num_positive_samples: int = 10
):
    """
    Plot C hyperparameter sweep for linear probes.
    
    Args:
        eval_dataset: Evaluation dataset name
        run_name: Model run name
        save_path: Path to save the plot
        experiment: Experiment type
        num_positive_samples: Number of positive samples to analyze
    """
    plot_size, _ = setup_plot_style()
    
    # Get data for linear probe C sweep
    df = get_hyperparameter_sweep_data(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        probe_type='sklearn_linear',
        hyperparam_name='C',
        num_positive_samples=num_positive_samples
    )
    
    if df.empty:
        print(f"No data found for linear C sweep: {eval_dataset}, {run_name}")
        return
    
    # Get unique C values
    c_values = sorted(df['C'].dropna().unique())
    
    if len(c_values) < 2:
        print(f"Insufficient C values for sweep: {c_values}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    # Calculate performance for each C value
    c_performances = []
    c_errors = []
    
    for c_val in c_values:
        c_data = df[df['C'] == c_val]
        if not c_data.empty:
            # Calculate mean and standard error across seeds
            auc_values = c_data['auc'].values
            mean_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
            sem_auc = std_auc / np.sqrt(len(auc_values))  # Standard error of mean
            
            c_performances.append(mean_auc)
            c_errors.append(sem_auc)
        else:
            c_performances.append(np.nan)
            c_errors.append(np.nan)
    
    # Plot with error bars
    ax.errorbar(c_values, c_performances, yerr=c_errors, 
               marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    # Formatting
    ax.set_xlabel('C (Regularization Parameter)')
    ax.set_ylabel('AUC')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    ax.set_title(f'{formatted_run_name} Linear Probes - C Sweep\n{formatted_dataset} ({num_positive_samples} positive samples)', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sae_probe_topk_sweep(
    eval_dataset: str,
    run_name: str,
    save_path: Path,
    experiment: str = '2-',
    num_positive_samples: int = 10
):
    """
    Plot topk hyperparameter sweep for SAE probes.
    
    Args:
        eval_dataset: Evaluation dataset name
        run_name: Model run name
        save_path: Path to save the plot
        experiment: Experiment type
        num_positive_samples: Number of positive samples to analyze
    """
    plot_size, _ = setup_plot_style()
    
    # Get data for SAE probe topk sweep
    df = get_hyperparameter_sweep_data(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        probe_type='sae',
        hyperparam_name='topk',
        num_positive_samples=num_positive_samples
    )
    
    if df.empty:
        print(f"No data found for SAE topk sweep: {eval_dataset}, {run_name}")
        return
    
    # Get unique topk values
    topk_values = sorted(df['topk'].dropna().unique())
    
    if len(topk_values) < 2:
        print(f"Insufficient topk values for sweep: {topk_values}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    # Calculate performance for each topk value
    topk_performances = []
    topk_errors = []
    
    for topk_val in topk_values:
        topk_data = df[df['topk'] == topk_val]
        if not topk_data.empty:
            # Calculate mean and standard error across seeds
            auc_values = topk_data['auc'].values
            mean_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
            sem_auc = std_auc / np.sqrt(len(auc_values))  # Standard error of mean
            
            topk_performances.append(mean_auc)
            topk_errors.append(sem_auc)
        else:
            topk_performances.append(np.nan)
            topk_errors.append(np.nan)
    
    # Plot with error bars
    ax.errorbar(topk_values, topk_performances, yerr=topk_errors, 
               marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    # Formatting
    ax.set_xlabel('Top-k')
    ax.set_ylabel('AUC')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    ax.set_title(f'{formatted_run_name} SAE Probes - Top-k Sweep\n{formatted_dataset} ({num_positive_samples} positive samples)', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_probe_weight_decay_sweep(
    eval_dataset: str,
    run_name: str,
    save_path: Path,
    experiment: str = '2-',
    num_positive_samples: int = 10
):
    """
    Plot weight decay hyperparameter sweep for attention probes at different learning rates.
    
    Args:
        eval_dataset: Evaluation dataset name
        run_name: Model run name
        save_path: Path to save the plot
        experiment: Experiment type
        num_positive_samples: Number of positive samples to analyze
    """
    plot_size, _ = setup_plot_style()
    
    # Get data for attention probe weight decay sweep
    df = get_hyperparameter_sweep_data(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        probe_type='attention',
        hyperparam_name='weight_decay',
        num_positive_samples=num_positive_samples
    )
    
    if df.empty:
        print(f"No data found for attention weight decay sweep: {eval_dataset}, {run_name}")
        return
    
    # Get unique learning rates
    lr_values = sorted(df['lr'].dropna().unique())
    wd_values = sorted(df['weight_decay'].dropna().unique())
    
    if len(wd_values) < 2:
        print(f"Insufficient weight decay values for sweep: {wd_values}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    # Plot each learning rate as a separate line
    colors = plt.cm.viridis(np.linspace(0, 1, len(lr_values)))
    
    for i, lr_val in enumerate(lr_values):
        lr_data = df[df['lr'] == lr_val]
        
        wd_performances = []
        wd_errors = []
        
        for wd_val in wd_values:
            wd_data = lr_data[lr_data['weight_decay'] == wd_val]
            if not wd_data.empty:
                # Calculate mean and standard error across seeds
                auc_values = wd_data['auc'].values
                mean_auc = np.mean(auc_values)
                std_auc = np.std(auc_values)
                sem_auc = std_auc / np.sqrt(len(auc_values))  # Standard error of mean
                
                wd_performances.append(mean_auc)
                wd_errors.append(sem_auc)
            else:
                wd_performances.append(np.nan)
                wd_errors.append(np.nan)
        
        # Plot with error bars
        ax.errorbar(wd_values, wd_performances, yerr=wd_errors, 
                   marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                   color=colors[i], label=f'lr={lr_val:.2e}')
    
    # Formatting
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('AUC')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    ax.set_title(f'{formatted_run_name} Attention Probes - Weight Decay Sweep\n{formatted_dataset} ({num_positive_samples} positive samples)', y=1.02)
    
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
    
    # Filter datasets (same as original system)
    def _leading_id(name: str) -> int:
        try:
            return int(str(name).split('_', 1)[0])
        except Exception:
            return -1
    
    eval_datasets = [d for d in eval_datasets if _leading_id(d) == -1 or _leading_id(d) < 99]
    
    # Define specific configurations for hyperparameter sweeps
    sweep_configs = [
        # Linear probe C sweeps
        {
            'type': 'linear_c',
            'run_names': ['spam_gemma_9b', 'mask_llama33_70b'],
            'eval_datasets': ['enron_spam'],
            'experiment': '2-',
            'num_positive_samples': 10
        },
        # SAE probe topk sweeps
        {
            'type': 'sae_topk',
            'run_names': ['mask_llama33_70b', 'spam_gemma_9b'],
            'eval_datasets': ['enron_spam'],
            'experiment': '2-',
            'num_positive_samples': 10
        },
        # Attention probe weight decay sweeps
        {
            'type': 'attention_wd',
            'run_names': ['spam_gemma_9b', 'mask_llama33_70b'],
            'eval_datasets': ['enron_spam'],
            'experiment': '2-',
            'num_positive_samples': 10
        }
    ]
    
    # Generate plots for each configuration
    for config in sweep_configs:
        sweep_type = config['type']
        
        for run_name in config['run_names']:
            if run_name not in run_names:
                print(f"Run {run_name} not found, skipping...")
                continue
                
            for eval_dataset in config['eval_datasets']:
                if eval_dataset not in eval_datasets:
                    print(f"Dataset {eval_dataset} not found, skipping...")
                    continue
                
                num_samples = config['num_positive_samples']
                experiment = config['experiment']
                
                # Generate plot based on type
                if sweep_type == 'linear_c':
                    save_path = output_dir / f"linear_c_sweep_{eval_dataset}_{run_name}_{num_samples}pos.png"
                    if not skip_existing or not save_path.exists():
                        plot_linear_probe_c_sweep(
                            eval_dataset, run_name, save_path, experiment, num_samples
                        )
                
                elif sweep_type == 'sae_topk':
                    save_path = output_dir / f"sae_topk_sweep_{eval_dataset}_{run_name}_{num_samples}pos.png"
                    if not skip_existing or not save_path.exists():
                        plot_sae_probe_topk_sweep(
                            eval_dataset, run_name, save_path, experiment, num_samples
                        )
                
                elif sweep_type == 'attention_wd':
                    save_path = output_dir / f"attention_wd_sweep_{eval_dataset}_{run_name}_{num_samples}pos.png"
                    if not skip_existing or not save_path.exists():
                        plot_attention_probe_weight_decay_sweep(
                            eval_dataset, run_name, save_path, experiment, num_samples
                        )
    
    print(f"Hyperparameter sweep plots generated in: {output_dir}")


if __name__ == "__main__":
    generate_all_hyperparameter_sweeps()
