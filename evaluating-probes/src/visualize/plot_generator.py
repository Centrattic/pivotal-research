import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from .data_loader import (
    get_data_for_visualization,
    get_probe_names,
    get_eval_datasets,
    get_run_names
)


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 4


def get_probe_label(probe_name: str) -> str:
    """Get human-readable label for probe names."""
    label_map = {
        'act_sim_last': 'Activation Similarity (last)',
        'act_sim_max': 'Activation Similarity (max)',
        'act_sim_mean': 'Activation Similarity (mean)',
        'act_sim_softmax': 'Activation Similarity (softmax)',
        'sae_last': 'SAE (last)',
        'sae_max': 'SAE (max)',
        'sae_mean': 'SAE (mean)',
        'sae_softmax': 'SAE (softmax)',
        'sklearn_linear_last': 'Linear (last)',
        'sklearn_linear_max': 'Linear (max)',
        'sklearn_linear_mean': 'Linear (mean)',
        'sklearn_linear_softmax': 'Linear (softmax)',
    }
    return label_map.get(probe_name, probe_name)


def wrap_text(text: str, max_length: int = 30) -> str:
    """Wrap text to avoid long labels."""
    if len(text) <= max_length:
        return text
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= max_length:
            current_line += (" " + word if current_line else word)
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def plot_experiment_2_best_probes(eval_dataset: str, save_path: Path):
    """Plot experiment 2 best probes comparison for a specific eval dataset."""
    setup_plot_style()
    
    # Get data for experiment 2
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Get unique probe names
    probe_names = df['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    
    # Calculate median AUC for each probe across seeds
    probe_medians = []
    probe_labels = []
    
    for probe in probe_names:
        probe_data = df[df['probe_name'] == probe]
        if not probe_data.empty:
            median_auc = probe_data['auc'].median()
            probe_medians.append(median_auc)
            probe_labels.append(get_probe_label(probe))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort by median AUC
    sorted_data = sorted(zip(probe_medians, probe_labels), reverse=True)
    medians, labels = zip(*sorted_data)
    
    bars = ax.bar(range(len(medians)), medians, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, median) in enumerate(zip(bars, medians)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{median:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Probe Type')
    ax.set_ylabel('Median AUC')
    ax.set_title(f'Experiment 2: Best Probes Comparison\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_text(label) for label in labels], rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_4_best_probes(eval_dataset: str, save_path: Path):
    """Plot experiment 4 best probes comparison for a specific eval dataset."""
    setup_plot_style()
    
    # Get data for experiment 4
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='4-',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Get unique probe names
    probe_names = df['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    
    # Calculate median AUC for each probe across seeds
    probe_medians = []
    probe_labels = []
    
    for probe in probe_names:
        probe_data = df[df['probe_name'] == probe]
        if not probe_data.empty:
            median_auc = probe_data['auc'].median()
            probe_medians.append(median_auc)
            probe_labels.append(get_probe_label(probe))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort by median AUC
    sorted_data = sorted(zip(probe_medians, probe_labels), reverse=True)
    medians, labels = zip(*sorted_data)
    
    bars = ax.bar(range(len(medians)), medians, color='lightgreen', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, median) in enumerate(zip(bars, medians)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{median:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Probe Type')
    ax.set_ylabel('Median AUC')
    ax.set_title(f'Experiment 4: Best Probes Comparison\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_text(label) for label in labels], rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_aggregated(eval_dataset: str, save_path: Path):
    """Plot aggregated LLM upsampling (median across all probes)."""
    setup_plot_style()
    
    # Get data for experiment 3 (LLM upsampling)
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling, eval_dataset={eval_dataset}")
        return
    
    # Extract upsampling factor from filename
    def extract_upsampling_factor(filename):
        import re
        match = re.search(r'_(\d+)x_', filename)
        return int(match.group(1)) if match else 1
    
    df['upsampling_factor'] = df['filename'].apply(extract_upsampling_factor)
    
    # Calculate median AUC for each upsampling factor across all probes
    upsampling_medians = []
    upsampling_factors = sorted(df['upsampling_factor'].unique())
    
    for factor in upsampling_factors:
        factor_data = df[df['upsampling_factor'] == factor]
        if not factor_data.empty:
            median_auc = factor_data['auc'].median()
            upsampling_medians.append(median_auc)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(upsampling_factors, upsampling_medians, 'o-', linewidth=2, markersize=6, color='purple')
    
    ax.set_xlabel('Upsampling Factor')
    ax.set_ylabel('Median AUC (across all probes)')
    ax.set_title(f'LLM Upsampling: Median Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_aggregated(eval_dataset: str, save_path: Path):
    """Plot aggregated scaling law (median across all probes)."""
    setup_plot_style()
    
    # Get Qwen run names for scaling analysis
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    qwen_runs.sort(key=lambda x: float(x.split('_')[-1].replace('b', '').replace('B', '')))
    
    # Get data for experiment 2 across all Qwen models
    all_data = []
    model_sizes = []
    
    for run in qwen_runs:
        df = get_data_for_visualization(
            eval_dataset=eval_dataset,
            experiment='2-',
            run_name=run,
            exclude_attention=True
        )
        
        if not df.empty:
            median_auc = df['auc'].median()
            all_data.append(median_auc)
            
            # Extract model size
            size_str = run.split('_')[-1].replace('b', '').replace('B', '')
            model_sizes.append(float(size_str))
    
    if not all_data:
        print(f"No data found for scaling law, eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(model_sizes, all_data, 'o-', linewidth=2, markersize=6, color='orange')
    
    ax.set_xlabel('Model Size (B parameters)')
    ax.set_ylabel('Median AUC (across all probes)')
    ax.set_title(f'Scaling Law: Median Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_ylim(0, 1.0)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probe_subplots(experiment: str, eval_dataset: str, save_path: Path):
    """Create subplot visualization for all probes (4x3 grid, excluding attention)."""
    setup_plot_style()
    
    # Get data for the experiment
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment=experiment,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for experiment {experiment}, eval_dataset={eval_dataset}")
        return
    
    # Get unique probe names
    probe_names = df['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    probe_names.sort()
    
    # Create 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, probe in enumerate(probe_names[:12]):  # Limit to 12 probes
        ax = axes[i]
        
        # Get data for this probe
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty:
            # Extract training set size from filename (for experiments 2 and 4)
            def extract_train_size(filename):
                import re
                # Look for patterns like class1_100, class1_500, etc.
                match = re.search(r'class1_(\d+)', filename)
                return int(match.group(1)) if match else 1000  # Default
            
            probe_data['train_size'] = probe_data['filename'].apply(extract_train_size)
            
            # Sort by training size
            probe_data = probe_data.sort_values('train_size')
            
            # Plot AUC vs training size
            ax.plot(probe_data['train_size'], probe_data['auc'], 'o-', markersize=4)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Training Size')
            ax.set_ylabel('AUC')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
    
    # Hide unused subplots
    for i in range(len(probe_names), 12):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Experiment {experiment}: Individual Probe Performance\nEval Dataset: {wrap_text(eval_dataset)}', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_subplots(eval_dataset: str, save_path: Path):
    """Create subplot visualization for LLM upsampling across all probes."""
    setup_plot_style()
    
    # Get data for experiment 3
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling, eval_dataset={eval_dataset}")
        return
    
    # Extract upsampling factor
    def extract_upsampling_factor(filename):
        import re
        match = re.search(r'_(\d+)x_', filename)
        return int(match.group(1)) if match else 1
    
    df['upsampling_factor'] = df['filename'].apply(extract_upsampling_factor)
    
    # Get unique probe names
    probe_names = df['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    probe_names.sort()
    
    # Create 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, probe in enumerate(probe_names[:12]):  # Limit to 12 probes
        ax = axes[i]
        
        # Get data for this probe
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty:
            # Group by upsampling factor and calculate median
            grouped = probe_data.groupby('upsampling_factor')['auc'].median().reset_index()
            grouped = grouped.sort_values('upsampling_factor')
            
            ax.plot(grouped['upsampling_factor'], grouped['auc'], 'o-', markersize=4)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Upsampling Factor')
            ax.set_ylabel('Median AUC')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
    
    # Hide unused subplots
    for i in range(len(probe_names), 12):
        axes[i].set_visible(False)
    
    plt.suptitle(f'LLM Upsampling: Individual Probe Performance\nEval Dataset: {wrap_text(eval_dataset)}', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_subplots(eval_dataset: str, save_path: Path):
    """Create subplot visualization for scaling law across all probes."""
    setup_plot_style()
    
    # Get Qwen run names
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    qwen_runs.sort(key=lambda x: float(x.split('_')[-1].replace('b', '').replace('B', '')))
    
    # Get unique probe names from any Qwen run
    df_sample = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        run_name=qwen_runs[0] if qwen_runs else None,
        exclude_attention=True
    )
    
    probe_names = df_sample['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    probe_names.sort()
    
    # Create 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, probe in enumerate(probe_names[:12]):  # Limit to 12 probes
        ax = axes[i]
        
        # Collect data for this probe across all Qwen models
        model_sizes = []
        auc_values = []
        
        for run in qwen_runs:
            df = get_data_for_visualization(
                eval_dataset=eval_dataset,
                experiment='2-',
                run_name=run,
                probe_name=probe
            )
            
            if not df.empty:
                median_auc = df['auc'].median()
                auc_values.append(median_auc)
                
                # Extract model size
                size_str = run.split('_')[-1].replace('b', '').replace('B', '')
                model_sizes.append(float(size_str))
        
        if model_sizes and auc_values:
            ax.plot(model_sizes, auc_values, 'o-', markersize=4)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Model Size (B)')
            ax.set_ylabel('Median AUC')
            ax.set_ylim(0, 1.0)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
    
    # Hide unused subplots
    for i in range(len(probe_names), 12):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Scaling Law: Individual Probe Performance\nEval Dataset: {wrap_text(eval_dataset)}', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_visualizations():
    """Generate all visualizations and save them to appropriate directories."""
    # Create output directories
    main_dir = Path("visualizations/main")
    other_dir = Path("visualizations/other")
    main_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available eval datasets
    eval_datasets = get_eval_datasets()
    
    print("Generating main visualizations...")
    
    # Generate main plots (experiment 2 and 4 best probes)
    for eval_dataset in eval_datasets:
        # Experiment 2 best probes
        plot_experiment_2_best_probes(
            eval_dataset, 
            main_dir / f"experiment_2_best_probes_{eval_dataset}.png"
        )
        
        # Experiment 4 best probes
        plot_experiment_4_best_probes(
            eval_dataset, 
            main_dir / f"experiment_4_best_probes_{eval_dataset}.png"
        )
        
        # LLM upsampling aggregated
        plot_llm_upsampling_aggregated(
            eval_dataset, 
            main_dir / f"llm_upsampling_aggregated_{eval_dataset}.png"
        )
        
        # Scaling law aggregated
        plot_scaling_law_aggregated(
            eval_dataset, 
            main_dir / f"scaling_law_aggregated_{eval_dataset}.png"
        )
    
    print("Generating other visualizations (subplots)...")
    
    # Generate subplot visualizations
    for eval_dataset in eval_datasets:
        # Experiment 2 subplots
        plot_probe_subplots(
            '2-', eval_dataset, 
            other_dir / f"experiment_2_subplots_{eval_dataset}.png"
        )
        
        # Experiment 4 subplots
        plot_probe_subplots(
            '4-', eval_dataset, 
            other_dir / f"experiment_4_subplots_{eval_dataset}.png"
        )
        
        # LLM upsampling subplots
        plot_llm_upsampling_subplots(
            eval_dataset, 
            other_dir / f"llm_upsampling_subplots_{eval_dataset}.png"
        )
        
        # Scaling law subplots
        plot_scaling_law_subplots(
            eval_dataset, 
            other_dir / f"scaling_law_subplots_{eval_dataset}.png"
        )
    
    print("All visualizations generated successfully!")
    print(f"Main plots saved to: {main_dir}")
    print(f"Other plots saved to: {other_dir}")


if __name__ == "__main__":
    generate_all_visualizations()
