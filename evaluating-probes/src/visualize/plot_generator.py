import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from scipy import stats
import re
from .data_loader import (
    get_data_for_visualization,
    get_probe_names,
    get_eval_datasets,
    get_run_names,
    filter_gemma_sae_topk_1024,
    filter_linear_probes_c_1_0,
    filter_default_attention_probes
)
from .viz_util import (
    setup_plot_style,
    get_probe_label,
    get_detailed_probe_label,
    calculate_confidence_interval,
    format_run_name_for_display,
    format_dataset_name_for_display,
    wrap_text,
    add_clean_log_inset,
    get_best_probes_by_category,
    apply_main_plot_filters,
    extract_model_size,
    plot_experiment_best_probes_generic,
    plot_probe_subplots_generic
)

# All these functions are now imported from viz_util.py

def plot_llm_upsampling_aggregated(eval_dataset: str, save_path: Path, metric: str = 'auc',):
    """Plot LLM upsampling with median performance across all probes."""
    plot_size, _= setup_plot_style()
    
    # Get data for experiment 3 (LLM upsampling)
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling, eval_dataset={eval_dataset}")
        return
    
    # Get unique upsampling ratios
    upsampling_ratios = sorted(df['llm_upsampling_ratio'].dropna().unique())
    
    if not upsampling_ratios:
        print(f"No upsampling ratios found for eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
    all_y_values = []
    lines_plotted = 0
    
    for i, ratio in enumerate(upsampling_ratios):
        ratio_data = df[df['llm_upsampling_ratio'] == ratio]
        
        if ratio_data.empty:
            continue
        
        # Group by number of positive samples and calculate median across all probes
        grouped = ratio_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(metric_list) for metric_list in grouped[metric]]
        all_y_values.extend(y_medians)
        
        # Plot line with cleaner label (e.g., "1x" instead of "1.0x")
        ratio_int = int(ratio) if ratio.is_integer() else ratio
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{ratio_int}x', linewidth=4, markersize=8)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for LLM upsampling {metric}, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.02)  # Reduced from 0.05 to 0.02 to get closer to smallest point
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
        title_suffix = 'AUC'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01\n(across all probes)'
        title_suffix = 'Recall'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    # Format eval_dataset for cleaner title
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel, labelpad=15)  # Add padding to y-axis label
    # Special handling for 87_is_spam dataset (OOD generalization)
    if '87_is_spam' in eval_dataset:
        ax.set_title(f'OOD Generalization: LLM-Upsampled Gemma-2-9b Probes\nEvaluated On SMS Spam Dataset', y=1.02)
    else:
        ax.set_title(f'LLM Upsampling: Median {title_suffix} Performance\n{formatted_dataset} Dataset', y=1.02)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend(title='Upsampling \nFactor')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_aggregated(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Plot scaling law with median performance across all probes."""
    plot_size, _ = setup_plot_style()
    
    # Get Qwen run names for scaling analysis
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    
    # Sort by model size (extract size and sort numerically)
    def extract_model_size(run_name):
        size_str = run_name.split('_')[-1].replace('b', '').replace('B', '')
        return float(size_str)
    
    qwen_runs.sort(key=extract_model_size)
    
    if not qwen_runs:
        print(f"No Qwen runs found for scaling law, eval_dataset={eval_dataset}")
        return
    
    # Debug: print the sorted runs to verify ordering
    print(f"Scaling law runs (sorted by size): {qwen_runs}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(qwen_runs)))
    all_y_values = []
    lines_plotted = 0
    
    for i, run in enumerate(qwen_runs):
        df = get_data_for_visualization(
            eval_dataset=eval_dataset,
            experiment='2-',
            run_name=run,
            exclude_attention=True
        )
        
        if df.empty:
            print(f"No data found for {run}")
            continue
        
        # Extract model size for legend
        size_str = run.split('_')[-1].replace('b', '').replace('B', '')
        model_size = float(size_str)
        
        print(f"Processing {run} (size: {model_size}B) with {len(df)} data points")
        
        # Group by number of positive samples and calculate median across all probes
        grouped = df.groupby('num_positive_samples')[metric].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            print(f"No grouped data for {run}")
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(metric_list) for metric_list in grouped[metric]]
        all_y_values.extend(y_medians)
        
        print(f"  {run}: x_values={x_values}, y_medians={y_medians}")
        
        # Plot line
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{model_size}B', linewidth=4, markersize=8)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for scaling law {metric}, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data with linear scale
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01\n(across all probes)'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    
    # Set title with OOD generalization for 87_is_spam dataset
    if '87_is_spam' in eval_dataset:
        ax.set_title(f'OOD Generalization: Scaling Probes Across Qwen-3 Model Sizes\nAnd Evaluating On SMS Spam', y=1.02)
    else:
        ax.set_title(f'Scaling Probes Across Qwen-3 Model Sizes on Enron-Spam', y=1.02)
    
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend(title='Qwen-3 Size')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_visualizations(skip_existing: bool = False):
    """Generate all visualizations and save them to appropriate directories.
    
    Args:
        skip_existing: If True, skip generating plots that already exist
    """
    # Create output directories
    main_dir = Path("visualizations/main")
    other_dir = Path("visualizations/other")
    main_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available eval datasets and run names
    eval_datasets = get_eval_datasets()
    run_names = get_run_names()
    
    # Filter: ignore datasets with numeric ID >= 99 (e.g., 99_*, 100_*)
    def _leading_id(name: str) -> int:
        try:
            return int(str(name).split('_', 1)[0])
        except Exception:
            return -1  # keep if it doesn't start with a number
    original = list(eval_datasets)
    eval_datasets = [d for d in eval_datasets if _leading_id(d) == -1 or _leading_id(d) < 99]
    dropped = sorted(set(original) - set(eval_datasets))
    if dropped:
        print(f"[viz] Skipping eval_datasets with ID >= 99: {dropped}")
    
    # Filter run_names to only include Gemma and MASK models (remove Qwen)
    gemma_mask_models = [name for name in run_names if 'gemma' in name.lower() or 'mask' in name.lower()]
    print(f"Generating visualizations for {len(gemma_mask_models)} Gemma/MASK models: {gemma_mask_models}")
    
    # Generate plots for each Gemma/MASK model separately
    for run_name in gemma_mask_models:
        print(f"\nProcessing model: {run_name}")
        
        for eval_dataset in eval_datasets:
            # Generate main plots (experiment 2 and 4 best probes) for this model
            # Experiment 2 best probes - AUC
            save_path = main_dir / f"experiment_2_best_probes_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_2_best_probes_for_model(eval_dataset, run_name, save_path, metric='auc')
            
            # Experiment 2 best probes - Recall
            save_path = main_dir / f"experiment_2_best_probes_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_2_best_probes_for_model(eval_dataset, run_name, save_path, metric='recall')
            
            # Experiment 4 best probes - AUC
            save_path = main_dir / f"experiment_4_best_probes_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_4_best_probes_for_model(eval_dataset, run_name, save_path, metric='auc')
            
            # Experiment 4 best probes - Recall
            save_path = main_dir / f"experiment_4_best_probes_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_4_best_probes_for_model(eval_dataset, run_name, save_path, metric='recall')
            
            # Joint best probes comparison - AUC
            save_path = main_dir / f"joint_best_probes_comparison_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_joint_best_probes_comparison_auc(eval_dataset, run_name, save_path)
            
            # Joint best probes comparison - Recall
            save_path = main_dir / f"joint_best_probes_comparison_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_joint_best_probes_comparison_recall(eval_dataset, run_name, save_path)
            
            # Note: LLM upsampling aggregated plots are generated later for all models together
    
    print("\nGenerating subplot visualizations for Gemma/MASK models...")
    
    # Generate subplot visualizations for each Gemma/MASK model
    for run_name in gemma_mask_models:
        print(f"\nProcessing subplots for model: {run_name}")
        
        for eval_dataset in eval_datasets:
            # Experiment 2 subplots
            save_path = main_dir / f"experiment_2_subplots_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_unified_for_model('2-', eval_dataset, run_name, save_path, metric='auc')
            
            # Experiment 2 subplots
            save_path = main_dir / f"experiment_2_subplots_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_unified_for_model('2-', eval_dataset, run_name, save_path, metric='recall')
            
            # Experiment 4 subplots
            save_path = main_dir / f"experiment_4_subplots_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_unified_for_model('4-', eval_dataset, run_name, save_path, metric='auc')
            
            # Experiment 4 subplots
            save_path = main_dir / f"experiment_4_subplots_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_unified_for_model('4-', eval_dataset, run_name, save_path, metric='recall')
            
            # LLM upsampling subplots
            save_path = main_dir / f"llm_upsampling_subplots_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_subplots_unified_for_model(eval_dataset, run_name, save_path, metric='auc')
            
            # LLM upsampling subplots
            save_path = main_dir / f"llm_upsampling_subplots_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_subplots_unified_for_model(eval_dataset, run_name, save_path, metric='recall')
    
    print("\nGenerating main aggregated plots...")
    
    # Generate main aggregated plots (all models together)
    for eval_dataset in eval_datasets:
        # LLM upsampling aggregated (Gemma only) - AUC
        save_path = main_dir / f"llm_upsampling_aggregated_auc_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_llm_upsampling_aggregated(eval_dataset, save_path, metric='auc')
        
        # LLM upsampling aggregated (Gemma only) - Recall
        save_path = main_dir / f"llm_upsampling_aggregated_recall_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_llm_upsampling_aggregated(eval_dataset, save_path, metric='recall')
        
        # Scaling law aggregated (Qwen only) - AUC
        save_path = main_dir / f"scaling_law_aggregated_auc_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_aggregated(eval_dataset, save_path, metric='auc')
        
        # Scaling law aggregated (Qwen only) - Recall
        save_path = main_dir / f"scaling_law_aggregated_recall_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_aggregated(eval_dataset, save_path, metric='recall')
    
    print("\nGenerating scaling law visualizations...")
    
    # Generate scaling law subplots (2 figures: one for AUC, one for Recall)
    for eval_dataset in eval_datasets:
        # Scaling law subplots - AUC (all models together)
        save_path = main_dir / f"scaling_law_subplots_auc_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_subplots_all_models(eval_dataset, save_path, metric='auc')
        
        # Scaling law subplots - Recall (all models together)
        save_path = main_dir / f"scaling_law_subplots_recall_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_subplots_all_models(eval_dataset, save_path, metric='recall')
    
    print("\nGenerating heatmap visualizations...")
    
    # Generate improved scaling law heatmaps (probes by model size, averaged over 1-20 samples)
    for eval_dataset in eval_datasets:
        # Scaling Law Heatmap - AUC (probes by model size)
        save_path = main_dir / f"scaling_law_heatmap_auc_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_heatmap_probes_by_model(eval_dataset, save_path, metric='auc')
        
        # Scaling Law Heatmap - Recall (probes by model size)
        save_path = main_dir / f"scaling_law_heatmap_recall_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_heatmap_probes_by_model(eval_dataset, save_path, metric='recall')
    
    # Generate LLM upsampling heatmaps (probes by upsampling factors, averaged over 1-10 samples)
    for eval_dataset in eval_datasets:
        # LLM Upsampling Heatmap - AUC
        save_path = main_dir / f"llm_upsampling_heatmap_auc_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_llm_upsampling_heatmap_probes_by_ratio(eval_dataset, save_path, metric='auc')
        
        # LLM Upsampling Heatmap - Recall
        save_path = main_dir / f"llm_upsampling_heatmap_recall_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_llm_upsampling_heatmap_probes_by_ratio(eval_dataset, save_path, metric='recall')
    
    print("All visualizations generated successfully!")
    print(f"Main plots (Gemma/MASK) saved to: {main_dir}")

def plot_experiment_2_best_probes_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Plot experiment 2 best probes as line chart with confidence intervals for a specific model."""
    # Use the shared generic function with default hyperparameter filters (None)
    plot_experiment_best_probes_generic(
        eval_dataset=eval_dataset,
        run_name=run_name,
        save_path=save_path,
        experiment='2-',
        metric=metric,
        hyperparam_filters=None,  # Use default filters
        title_suffix="",
        output_dir=None  # Save to original location
    )


# These functions are now imported from viz_util.py





def plot_experiment_4_best_probes_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Plot experiment 4 best probes as line chart with confidence intervals for a specific model."""
    # Use the shared generic function with default hyperparameter filters (None)
    plot_experiment_best_probes_generic(
        eval_dataset=eval_dataset,
        run_name=run_name,
        save_path=save_path,
        experiment='4-',
        metric=metric,
        hyperparam_filters=None,  # Use default filters
        title_suffix="",
        output_dir=None  # Save to original location
    )


# Placeholder functions for other plot types (to be implemented later)
def plot_llm_upsampling_aggregated_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Plot LLM upsampling with median performance across all probes for Gemma models only."""
    plot_size, _ = setup_plot_style()
    
    # Only generate for Gemma models
    if 'gemma' not in run_name.lower():
        print(f"Skipping LLM upsampling aggregated {metric} for {run_name} - only for Gemma models")
        return
    
    # Get data for experiment 3 (LLM upsampling) for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Get unique upsampling ratios
    upsampling_ratios = sorted(df['llm_upsampling_ratio'].dropna().unique())
    
    if not upsampling_ratios:
        print(f"No upsampling ratios found for eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
    all_y_values = []
    lines_plotted = 0
    
    for i, ratio in enumerate(upsampling_ratios):
        ratio_data = df[df['llm_upsampling_ratio'] == ratio]
        
        if ratio_data.empty:
            continue
        
        # Group by number of positive samples and calculate median across all probes
        grouped = ratio_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(metric_list) for metric_list in grouped[metric]]
        all_y_values.extend(y_medians)
        
        # Plot line with cleaner label (e.g., "1x" instead of "1.0x")
        ratio_int = int(ratio) if ratio.is_integer() else ratio
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{ratio_int}x', linewidth=4, markersize=8)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for LLM upsampling {metric}, eval_dataset={eval_dataset}, run_name={run_name}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.02)  # Reduced from 0.05 to 0.02 to get closer to smallest point
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
        title_suffix = 'AUC'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01\n(across all probes)'
        title_suffix = 'Recall'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    # Format run_name and eval_dataset for cleaner title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    # Special handling for 87_is_spam dataset (OOD generalization)
    if '87_is_spam' in eval_dataset:
        ax.set_title(f'OOD Generalization: LLM-Upsampled {formatted_run_name} Probes On SMS Spam Dataset', y=1.02)
    else:
        ax.set_title(f'{formatted_run_name} Probes on LLM-Upsampled {formatted_dataset}', y=1.02)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend(title='Upsampling \nFactor')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scaling_law_aggregated_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc',):
    """Plot scaling law with median performance across all probes for Qwen models only."""
    plot_size, _ = setup_plot_style()
    
    # Only generate for Qwen models
    if 'qwen' not in run_name.lower():
        print(f"Skipping scaling law aggregated {metric} for {run_name} - only for Qwen models")
        return
    
    # Get data for experiment 2 for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for scaling law, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    # Group by number of positive samples and calculate median across all probes
    grouped = df.groupby('num_positive_samples')[metric].apply(list).reset_index()
    grouped = grouped.sort_values('num_positive_samples')
    
    if grouped.empty:
        print(f"No data to plot for scaling law {metric}, eval_dataset={eval_dataset}, run_name={run_name}")
        plt.close()
        return
    
    x_values = grouped['num_positive_samples'].values
    y_medians = [np.median(metric_list) for metric_list in grouped[metric]]
    
    # Extract model size for legend
    size_str = run_name.split('_')[-1].replace('b', '').replace('B', '')
    model_size = float(size_str)
    
    print(f"Model-specific scaling law for {run_name} (size: {model_size}B): x_values={x_values}, y_medians={y_medians}")
    
    # Plot line
    ax.plot(x_values, y_medians, 'o-', color='blue', 
            label=f'{model_size}B', linewidth=4, markersize=8)
    
    # Set proper ylim based on actual data with linear scale
    if y_medians:
        y_min = max(0.0, min(y_medians) - 0.05)
        y_max = min(1.0, max(y_medians) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01\n(across all probes)'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Scaling Probes Across Qwen-3 Model Sizes on Enron-Spam - {run_name}', y=1.02)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(title='Qwen-3 Size')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probe_subplots_unified_for_model(experiment: str, eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create subplot visualization for all probes for a specific model - unified function for AUC and Recall."""
    # Use the shared generic function with default hyperparameter filters (None)
    plot_probe_subplots_generic(
        experiment=experiment,
        eval_dataset=eval_dataset,
        run_name=run_name,
        save_path=save_path,
        metric=metric,
        hyperparam_filters=None,  # Use default filters
        title_suffix="",
        output_dir=None  # Save to original location
    )





def plot_llm_upsampling_subplots_unified_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create LLM upsampling subplot visualization for all probes for a specific model - unified function for AUC and Recall."""
    _, plot_size= setup_plot_style()
    
    # Only generate for Gemma models
    if 'gemma' not in run_name.lower():
        print(f"Skipping LLM upsampling subplots for {run_name} - only for Gemma models")
        return
    
    # Get data for experiment 3 (LLM upsampling) for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Define the probe names we want to include
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
    ]
    
    # Create subplot grid based on number of probes
    fig, axes = plt.subplots(3, 4, figsize=plot_size)
    axes = axes.flatten()
    
    # Define colors for each probe type
    act_sim_color = 'blue'
    sae_color = 'orange'
    linear_color = 'green'
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'AUC'
        y_min_default = 0.6
        title_suffix = '(AUC)'
    elif metric == 'recall':
        ylabel = 'Recall @ FPR=0.01'
        y_min_default = 0.0
        title_suffix = '(Recall @ FPR=0.01)'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(probe_names):
        ax = axes[i]
        
        # Get data for this probe
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty and probe_data['num_positive_samples'].notna().any():
            # Group by number of positive samples and upsampling ratio
            grouped = probe_data.groupby(['num_positive_samples', 'llm_upsampling_ratio'])[metric].apply(list).reset_index()
            grouped = grouped.sort_values(['num_positive_samples', 'llm_upsampling_ratio'])
            
            # Get unique upsampling ratios
            upsampling_ratios = sorted(probe_data['llm_upsampling_ratio'].dropna().unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
            
            for j, ratio in enumerate(upsampling_ratios):
                ratio_data = grouped[grouped['llm_upsampling_ratio'] == ratio]
                if not ratio_data.empty:
                    x_values = ratio_data['num_positive_samples'].values
                    y_means = [np.mean(values) for values in ratio_data[metric]]
                    y_lower = []
                    y_upper = []
                    
                    for values in ratio_data[metric]:
                        lower, upper = calculate_confidence_interval(values)
                        y_lower.append(lower)
                        y_upper.append(upper)
                    
                    all_y_values.extend(y_means)
                    
                    # Use color based on upsampling ratio (not probe type)
                    color = colors[j]
                    
                    # Plot line with confidence interval
                    ax.plot(x_values, y_means, 'o-', color=color, markersize=4, linewidth=2, alpha=0.8)
                    ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=color)
            
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
            ax.set_xlabel('Num Positive Samples')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
    
    # Set consistent ylim across all subplots within this figure
    all_figure_y_values = []
    for probe in probe_names:
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty and probe_data['num_positive_samples'].notna().any():
            # Get data for this specific probe
            grouped = probe_data.groupby(['num_positive_samples', 'llm_upsampling_ratio'])[metric].apply(list).reset_index()
            grouped = grouped.sort_values(['num_positive_samples', 'llm_upsampling_ratio'])
            
            if not grouped.empty:
                for values in grouped[metric]:
                    all_figure_y_values.extend(values)
    
    # Calculate y-axis limits for all subplots in this figure
    if all_figure_y_values:
        # Use a more reasonable lower bound that provides better visibility
        # For AUC plots, use 0.6 as minimum, for recall plots use 0.0
        if metric == 'auc':
            # Special handling for 87_is_spam dataset - use lower bound based on actual data
            if '87_is_spam' in eval_dataset:
                y_min = max(0.5, min(all_figure_y_values) - 0.05)
            else:
                y_min = max(0.6, min(all_figure_y_values) - 0.05)
        else:  # recall
            y_min = max(0.0, min(all_figure_y_values) - 0.05)
        
        # Calculate upper bound more adaptively based on actual data
        actual_max = max(all_figure_y_values)
        if metric == 'auc':
            # For AUC, if the actual max is below 0.9, use a more reasonable upper bound
            if actual_max < 0.9:
                y_max = min(1.0, actual_max + 0.1)  # Add 0.1 instead of 0.05 for more space
            else:
                y_max = min(1.0, actual_max + 0.05)
        else:  # recall
            y_max = min(1.0, actual_max + 0.05)
        
        # Ensure y_min < y_max to prevent inverted axes
        if y_min < y_max:
            for ax in axes:
                ax.set_ylim(y_min, y_max)
        else:
            # Fallback to reasonable defaults if data is problematic
            for ax in axes:
                ax.set_ylim(0.6 if metric == 'auc' else 0.0, 1.0)
    else:
        # No valid y values, set reasonable defaults
        for ax in axes:
            ax.set_ylim(0.6 if metric == 'auc' else 0.0, 1.0)
    
    # Add legend at the bottom with better positioning and larger font
    legend_handles = []
    legend_labels = []
    upsampling_ratios = sorted(df['llm_upsampling_ratio'].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
    
    for j, ratio in enumerate(upsampling_ratios):
        line, = plt.plot([], [], 'o-', color=colors[j], markersize=6, linewidth=2, label=f'{int(ratio) if ratio.is_integer() else ratio}x')
        legend_handles.append(line)
        legend_labels.append(f'{int(ratio) if ratio.is_integer() else ratio}x')
    
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=len(legend_handles), fontsize=16, title='Upsampling Factor', title_fontsize=16)
    
    # Special handling for 87_is_spam dataset (OOD generalization)
    if '87_is_spam' in eval_dataset:
        fig.suptitle(f'OOD Generalization: LLM-Upsampled {format_run_name_for_display(run_name)} Probes\nEvaluated On SMS Spam Dataset', fontsize=20, y=0.98)
    else:
        fig.suptitle(f'LLM-Upsampled {format_run_name_for_display(run_name)} Probes On Enron-Spam Dataset', fontsize=20, y=0.98)
    
    plt.subplots_adjust(bottom=0.08)  # Greatly reduced bottom padding for legend
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_subplots_unified_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create scaling law subplot visualization for all probes for a specific model - unified function for AUC and Recall."""
    _, plot_size = setup_plot_style()
    
    # Only generate for Qwen models
    if 'qwen' not in run_name.lower():
        print(f"Skipping scaling law subplots for {run_name} - only for Qwen models")
        return
    
    # Get data for experiment 2 for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for scaling law, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Define the probe names we want to include
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
    ]
    
    # Create subplot grid based on number of probes
    fig, axes = plt.subplots(3, 4, figsize=plot_size)
    axes = axes.flatten()
    
    # Define colors for each probe type
    act_sim_color = 'blue'
    sae_color = 'orange'
    linear_color = 'green'
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'AUC'
        y_min_default = 0.6
        title_suffix = '(AUC)'
    elif metric == 'recall':
        ylabel = 'Recall @ FPR=0.01'
        y_min_default = 0.0
        title_suffix = '(Recall @ FPR=0.01)'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(probe_names):
        ax = axes[i]
        
        # Get data for this probe
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty and probe_data['num_positive_samples'].notna().any():
            # Group by number of positive samples
            grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
            grouped = grouped.sort_values('num_positive_samples')
            
            x_values = grouped['num_positive_samples'].values
            y_means = [np.mean(values) for values in grouped[metric]]
            all_y_values.extend(y_means)
            
            # Choose color based on probe type
            if 'act_sim' in probe:
                color = act_sim_color
            elif 'sae' in probe:
                color = sae_color
            elif 'linear' in probe:
                color = linear_color
            else:
                color = 'black'
            
            ax.plot(x_values, y_means, 'o-', color=color, markersize=8, linewidth=3)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
            ax.set_xlabel('Num Positive Samples')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
    
    # Set consistent ylim across all subplots within this figure
    all_figure_y_values = []
    for probe in probe_names:
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty and probe_data['num_positive_samples'].notna().any():
            # Get data for this specific probe
            grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
            grouped = grouped.sort_values('num_positive_samples')
            
            if not grouped.empty:
                for values in grouped[metric]:
                    all_figure_y_values.extend(values)
    
    # Calculate y-axis limits for all subplots in this figure
    if all_figure_y_values:
        # Use a more reasonable lower bound that provides better visibility
        # For AUC plots, use 0.6 as minimum, for recall plots use 0.0
        if metric == 'auc':
            y_min = max(0.6, min(all_figure_y_values) - 0.05)
        else:  # recall
            y_min = max(0.0, min(all_figure_y_values) - 0.05)
        y_max = min(1.0, max(all_figure_y_values) + 0.05)
        # Ensure y_min < y_max to prevent inverted axes
        if y_min < y_max:
            for ax in axes:
                ax.set_ylim(y_min, y_max)
        else:
            # Fallback to reasonable defaults if data is problematic
            for ax in axes:
                ax.set_ylim(0.6 if metric == 'auc' else 0.0, 1.0)
    else:
        # No valid y values, set reasonable defaults
        for ax in axes:
            ax.set_ylim(0.6 if metric == 'auc' else 0.0, 1.0)
    
    fig.suptitle(f'{run_name} - Scaling Law {title_suffix}\n{eval_dataset}', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc', plot_type: str = 'scaling_law'):
    """Create heatmap for a specific model and plot type.
    
    Args:
        eval_dataset: Evaluation dataset name
        run_name: Model run name
        save_path: Path to save the plot
        metric: 'auc' or 'recall'
        plot_type: 'scaling_law' or 'llm_upsampling'
    """
    _, plot_size = setup_plot_style()
    
    # Validate plot type and model compatibility
    if plot_type == 'scaling_law':
        if 'qwen' not in run_name.lower():
            print(f"Skipping scaling law heatmap for {run_name} - only for Qwen models")
            return
        experiment = '2-'
        columns = 'num_positive_samples'
        title_prefix = 'Scaling Law Heatmap'
        xlabel = 'Number of positive examples in the train set'
        figsize = plot_size
    elif plot_type == 'llm_upsampling':
        if 'gemma' not in run_name.lower():
            print(f"Skipping LLM upsampling heatmap for {run_name} - only for Gemma models")
            return
        experiment = '3-'
        columns = ['num_positive_samples', 'llm_upsampling_ratio']
        title_prefix = 'LLM Upsampling Heatmap'
        xlabel = '(Number of positive examples, Upsampling ratio)'
        figsize = (8, 4)
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}. Use 'scaling_law' or 'llm_upsampling'.")
    
    # Get data for the experiment for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for {plot_type} heatmap, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create heatmap data
    pivot_data = df.pivot_table(
        values=metric,
        index='probe_name',
        columns=columns,
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        print(f"No data to plot for {plot_type} heatmap, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    
    # Set metric-specific parameters
    if metric == 'auc':
        title_suffix = 'AUC'
    elif metric == 'recall':
        title_suffix = 'Recall @ FPR=0.01'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_title(f'{title_prefix}: {title_suffix}\n{run_name} - {eval_dataset}', y=1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probe Name')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_heatmap_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create scaling law heatmap for a specific model."""
    plot_heatmap_for_model(eval_dataset, run_name, save_path, metric, 'scaling_law')


def plot_llm_upsampling_heatmap_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create LLM upsampling heatmap for a specific model."""
    plot_heatmap_for_model(eval_dataset, run_name, save_path, metric, 'llm_upsampling')


def plot_joint_best_probes_comparison(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create a joint plot comparing best probes on unbalanced vs balanced datasets.
    
    This creates a side-by-side comparison with shared y-axis and legend.
    """
    plot_size, _ = setup_plot_style()
    
    # Only generate for Gemma and Mask models
    if not ('gemma' in run_name.lower() or 'mask' in run_name.lower()):
        print(f"Skipping joint best probes comparison for {run_name} - only for Gemma and Mask models")
        return
    
    # Get data for both experiments
    df_unbalanced = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        run_name=run_name,
        exclude_attention=False  # Include attention for best probes
    )
    
    df_balanced = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='4-',
        run_name=run_name,
        exclude_attention=False  # Include attention for best probes
    )
    
    if df_unbalanced.empty and df_balanced.empty:
        print(f"No data found for joint comparison, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Apply main plot filters to both datasets (this ensures only default attention probes are included)
    if not df_unbalanced.empty:
        df_unbalanced = apply_main_plot_filters(df_unbalanced)
    if not df_balanced.empty:
        df_balanced = apply_main_plot_filters(df_balanced)
    
    # Get best probes from combined dataset to ensure consistency across both plots
    df_combined = pd.concat([df_unbalanced, df_balanced], ignore_index=True)
    best_probes_combined = get_best_probes_by_category(df_combined) if not df_combined.empty else []
    
    # Use the same best probes for both plots to ensure consistency
    best_probes_unbalanced = best_probes_combined
    best_probes_balanced = best_probes_combined
    
    # Use the combined best probes for consistent colors
    all_probes = best_probes_combined
    if not all_probes:
        print(f"No valid probes found for joint comparison, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create subplot with shared y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    colors = ['blue', 'orange', 'green', 'red']
    all_y_values = []  # Collect all y values for proper ylim
    legend_handles = []  # Collect legend handles
    legend_labels = []  # Collect legend labels
    
    # Create a mapping from probe name to color for consistent coloring
    probe_color_map = {}
    for i, probe in enumerate(all_probes):
        probe_color_map[probe] = colors[i % len(colors)]
    
    # Plot unbalanced data (left subplot)
    if not df_unbalanced.empty and best_probes_unbalanced:
        for probe in best_probes_unbalanced:
            probe_data = df_unbalanced[df_unbalanced['probe_name'] == probe]
            
            if not probe_data.empty:
                # Group by number of positive samples
                grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
                grouped = grouped.sort_values('num_positive_samples')
                
                if not grouped.empty:
                    x_values = grouped['num_positive_samples'].values
                    y_means = [np.mean(metric_list) for metric_list in grouped[metric]]
                    y_lower = []
                    y_upper = []
                    
                    for metric_list in grouped[metric]:
                        lower, upper = calculate_confidence_interval(metric_list)
                        y_lower.append(lower)
                        y_upper.append(upper)
                    
                    # Collect all y values for ylim calculation
                    all_y_values.extend(y_means)
                    all_y_values.extend(y_lower)
                    all_y_values.extend(y_upper)
                    
                    # Plot line with confidence interval
                    color = probe_color_map[probe]
                    line, = ax1.plot(x_values, y_means, 'o-', color=color, 
                                   label=get_probe_label(probe), linewidth=4, markersize=8)
                    ax1.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=color)
                    
                    # Store legend handle and label
                    if get_probe_label(probe) not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(get_probe_label(probe))
    
    # Plot balanced data (right subplot)
    if not df_balanced.empty and best_probes_balanced:
        for probe in best_probes_balanced:
            probe_data = df_balanced[df_balanced['probe_name'] == probe]
            
            if not probe_data.empty:
                # Group by number of positive samples
                grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
                grouped = grouped.sort_values('num_positive_samples')
                
                if not grouped.empty:
                    x_values = grouped['num_positive_samples'].values
                    y_means = [np.mean(metric_list) for metric_list in grouped[metric]]
                    y_lower = []
                    y_upper = []
                    
                    for metric_list in grouped[metric]:
                        lower, upper = calculate_confidence_interval(metric_list)
                        y_lower.append(lower)
                        y_upper.append(upper)
                    
                    # Collect all y values for ylim calculation
                    all_y_values.extend(y_means)
                    all_y_values.extend(y_lower)
                    all_y_values.extend(y_upper)
                    
                    # Plot line with confidence interval (use same color as unbalanced)
                    color = probe_color_map[probe]
                    line, = ax2.plot(x_values, y_means, 'o-', color=color, 
                                   label=get_probe_label(probe), linewidth=4, markersize=8)
                    ax2.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=color)
                    
                    # Store legend handle and label
                    if get_probe_label(probe) not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(get_probe_label(probe))
    
    # Set proper ylim based on actual data with linear scale
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        # Create inset axes for zoomed view of values close to 1 (only for Gemma models with 94_better_spam dataset, AUC metric, and values > 0.9)
        if max(all_y_values) > 0.9 and ('gemma' in run_name.lower()) and ('94_better_spam' in eval_dataset) and metric == 'auc':
            axin1 = add_clean_log_inset(ax1, pos=(0.42, 0.10, 0.5, 0.4), xlim=(20, 120), ylim=(0.97, 1.00))
            axin2 = add_clean_log_inset(ax2, pos=(0.42, 0.10, 0.5, 0.4), xlim=(20, 120), ylim=(0.97, 1.00))
            
            # Re-plot the data in the inset with smaller markers
            for probe in best_probes_unbalanced:
                probe_data = df_unbalanced[df_unbalanced['probe_name'] == probe]
                
                if not probe_data.empty:
                    grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
                    grouped = grouped.sort_values('num_positive_samples')
                    
                    if not grouped.empty:
                        x_values = grouped['num_positive_samples'].values
                        y_means = [np.mean(metric_list) for metric_list in grouped[metric]]
                        
                        color = probe_color_map[probe]
                        axin1.plot(x_values, y_means, 'o-', color=color, 
                                 linewidth=2, markersize=4, alpha=0.9)
            
            # Re-plot the data in the inset with smaller markers
            for probe in best_probes_balanced:
                probe_data = df_balanced[df_balanced['probe_name'] == probe]
                
                if not probe_data.empty:
                    grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
                    grouped = grouped.sort_values('num_positive_samples')
                    
                    if not grouped.empty:
                        x_values = grouped['num_positive_samples'].values
                        y_means = [np.mean(metric_list) for metric_list in grouped[metric]]
                        
                        color = probe_color_map[probe]
                        axin2.plot(x_values, y_means, 'o-', color=color, 
                                 linewidth=2, markersize=4, alpha=0.9)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'AUC'
    elif metric == 'recall':
        ylabel = 'Recall @ FPR=0.01'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    # Format run_name for cleaner title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    
    # Get number of negative examples for unbalanced title
    num_negative = None
    if not df_unbalanced.empty and 'num_negative_samples' in df_unbalanced.columns:
        num_negative = df_unbalanced['num_negative_samples'].iloc[0]
    
    # Configure left subplot (unbalanced)
    ax1.set_xlabel('Number of positive examples in the train set')
    ax1.set_ylabel(ylabel)
    
    # Special handling for 87_is_spam dataset (OOD generalization)
    if '87_is_spam' in eval_dataset:
        ax1.set_title(f'OOD Generalization: Unbalanced {formatted_run_name} Probes\nEvaluated On SMS Spam Dataset', y=1.02)
    else:
        if num_negative is not None:
            # Convert to int to remove .0
            num_negative_int = int(num_negative)
            ax1.set_title(f'{formatted_run_name} Probes On Unbalanced {formatted_dataset} Dataset\n({num_negative_int} negative examples)', y=1.02)
        else:
            ax1.set_title(f'{formatted_run_name} Probes On Unbalanced {formatted_dataset} Dataset', y=1.02)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Configure right subplot (balanced)
    ax2.set_xlabel('Number of training examples per class\n(x negative, x positive)')
    ax2.set_ylabel('')  # No y-label for right subplot
    
    # Special handling for 87_is_spam dataset (OOD generalization)
    if '87_is_spam' in eval_dataset:
        ax2.set_title(f'OOD Generalization: Balanced {formatted_run_name} Probes\nEvaluated On SMS Spam Dataset', y=1.02)
    else:
        ax2.set_title(f'{formatted_run_name} Probes on Balanced {formatted_dataset} Dataset\n(equal positive and negative samples)', y=1.02)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add shared legend at the bottom with reduced padding
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=len(legend_handles), fontsize=18)
    
    # Adjust subplot parameters to reduce gap and add title padding
    plt.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_joint_best_probes_comparison_auc(eval_dataset: str, run_name: str, save_path: Path):
    """Create joint best probes comparison for AUC."""
    plot_joint_best_probes_comparison(eval_dataset, run_name, save_path, metric='auc')


def plot_joint_best_probes_comparison_recall(eval_dataset: str, run_name: str, save_path: Path):
    """Create joint best probes comparison for Recall."""
    plot_joint_best_probes_comparison(eval_dataset, run_name, save_path, metric='recall')


def plot_scaling_law_subplots_all_models(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Create scaling law subplot visualization showing all 6 scaling law lines per probe across all models."""
    _, plot_size = setup_plot_style()
    
    # Get data for scaling law experiment (2-spam-pred-auc-increasing-spam-fixed-total) for all models
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-spam-pred-auc-increasing-spam-fixed-total',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for scaling law experiment, eval_dataset={eval_dataset}")
        return
    
    # Get all run names (models) for scaling law
    run_names = sorted(df['run_name'].unique())
    
    # Define the probe names we want to include (12 probes total)
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
    ]
    
    # Create subplot grid: 3 rows x 4 columns for 12 probes
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    # Define colors for each probe type
    act_sim_color = 'blue'
    sae_color = 'orange'
    linear_color = 'green'
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'AUC'
        title_suffix = '(AUC)'
    elif metric == 'recall':
        ylabel = 'Recall @ FPR=0.01'
        title_suffix = '(Recall @ FPR=0.01)'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(probe_names):
        ax = axes[i]
        
        # Get data for this probe across all models
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty and probe_data['num_positive_samples'].notna().any():
            # Group by number of positive samples and run_name
            grouped = probe_data.groupby(['num_positive_samples', 'run_name'])[metric].apply(list).reset_index()
            grouped = grouped.sort_values(['num_positive_samples', 'run_name'])
            
            # Choose color based on probe type
            if 'act_sim' in probe:
                color = act_sim_color
            elif 'sae' in probe:
                color = sae_color
            elif 'linear' in probe:
                color = linear_color
            else:
                color = 'black'
            
            # Plot each model's scaling law line
            for run_name in run_names:
                model_data = grouped[grouped['run_name'] == run_name]
                if not model_data.empty:
                    x_values = model_data['num_positive_samples'].values
                    y_means = [np.mean(values) for values in model_data[metric]]
                    all_y_values.extend(y_means)
                    
                    # Format run name for display
                    formatted_run_name = format_run_name_for_display(run_name)
                    
                    # Plot line with confidence interval
                    ax.plot(x_values, y_means, 'o-', color=color, markersize=4, linewidth=1.5, alpha=0.7, label=formatted_run_name)
            
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
            ax.set_xlabel('Num Positive Samples')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='lower right')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
    
    # Set consistent ylim across all subplots
    if all_y_values:
        y_min = max(0.85, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    
    # Set main title
    fig.suptitle(f'Scaling Law Performance {title_suffix}\n{eval_dataset}', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_heatmap_probes_by_model(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Create scaling law heatmap showing probes by model size, averaged over 1-20 positive samples."""
    _, plot_size = setup_plot_style()
    
    # Get data for scaling law experiment (2-spam-pred-auc-increasing-spam-fixed-total)
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-spam-pred-auc-increasing-spam-fixed-total',
        exclude_attention=False
    )
    
    if df.empty:
        print(f"No data found for scaling law experiment, eval_dataset={eval_dataset}")
        return
    
    # Filter to only include 1-20 positive samples
    df_filtered = df[df['num_positive_samples'].between(1, 20)]
    
    if df_filtered.empty:
        print(f"No data found for 1-20 positive samples, eval_dataset={eval_dataset}")
        return
    
    # Define probe names (all 16 probes including attention)
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax',
        'attention_last', 'attention_max', 'attention_mean', 'attention_softmax'
    ]
    
    # Get run names, filter out Gemma models, and sort by model size
    all_run_names = df_filtered['run_name'].unique()
    run_names = [name for name in all_run_names if 'gemma' not in name.lower()]
    
    # Debug: Print run names and their extracted sizes
    print(f"Debug - Run names before sorting: {run_names}")
    for name in run_names:
        print(f"  {name} -> size {extract_model_size(name)}")
    
    run_names = sorted(run_names, key=lambda x: extract_model_size(x))
    
    print(f"Debug - Run names after sorting: {run_names}")
    
    # Create heatmap data
    heatmap_data = []
    probe_labels = []
    
    # Debug: Check what probe names are actually in the data
    available_probes = df_filtered['probe_name'].unique()
    # Filter out None values before sorting
    available_probes_clean = [p for p in available_probes if p is not None]
    print(f"Debug - Available probe names in data: {sorted(available_probes_clean)}")
    print(f"Debug - Looking for attention probes: {[p for p in probe_names if 'attention' in p]}")
    
    for probe in probe_names:
        probe_data = df_filtered[df_filtered['probe_name'] == probe]
        print(f"Debug - Probe '{probe}': {len(probe_data)} rows found")
        if not probe_data.empty:
            # Average across the 1-20 positive samples range for each model
            model_means = []
            for run_name in run_names:
                model_probe_data = probe_data[probe_data['run_name'] == run_name]
                if not model_probe_data.empty:
                    mean_value = model_probe_data[metric].mean()
                    model_means.append(mean_value)
                else:
                    model_means.append(np.nan)
            
            heatmap_data.append(model_means)
            probe_labels.append(get_detailed_probe_label(probe))
        else:
            print(f"Debug - No data found for probe '{probe}'")
    
    if not heatmap_data:
        print(f"No data to plot for heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap with smaller cells for 16 probes
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    im = ax.imshow(heatmap_array, cmap='viridis', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(run_names)))
    ax.set_xticklabels([format_run_name_for_display(name) for name in run_names], rotation=45, ha='right')
    ax.set_yticks(range(len(probe_labels)))
    ax.set_yticklabels(probe_labels)
    
    # Add x-axis label
    ax.set_xlabel('Model Size', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)

    if metric == 'recall':
        cbar.set_label('Recall @ FPR=0.01 (Averaged Over 1-20 Samples)', rotation=270, labelpad=20)
    else:
        cbar.set_label(metric.upper() + ' (Averaged Over 1-20 Samples)', rotation=270, labelpad=20)

    
    # Add text annotations with dynamic coloring
    for i in range(len(probe_labels)):
        for j in range(len(run_names)):
            if not np.isnan(heatmap_array[i, j]):
                # Determine text color based on background brightness
                value = heatmap_array[i, j]
                # Use white text for dark backgrounds, black for light backgrounds
                text_color = 'white' if value < 0.8 else 'black'
                text = ax.text(j, i, f'{value:.2f}', 
                             ha='center', va='center', color=text_color, fontsize=14,)
    
    # Set title with better description
    title_suffix = '(AUC)' if metric == 'auc' else '(Recall @ FPR=0.01)'
    if '87_is_spam' in eval_dataset:
        title = f'OOD Generalization: Qwen-3 Probes Evaluated On SMS Spam {title_suffix}'
    else:
        title = f'Qwen-3 Probes Trained And Evaluated on Enron-Spam {title_suffix}'
    ax.set_title(title, fontsize=20, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_heatmap_probes_by_ratio(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Create LLM upsampling heatmap showing probes by upsampling factors, averaged over 1-10 samples."""
    _, plot_size = setup_plot_style()
    
    # Get data for LLM upsampling experiment (experiment 3)
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        exclude_attention=False
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling experiment, eval_dataset={eval_dataset}")
        return
    
    # Filter to only include 1-10 positive samples
    df_filtered = df[df['num_positive_samples'].between(1, 10)]
    
    if df_filtered.empty:
        print(f"No data found for 1-10 positive samples, eval_dataset={eval_dataset}")
        return
    
    # Define probe names (all 16 probes including attention)
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax',
        'attention_last', 'attention_max', 'attention_mean', 'attention_softmax'
    ]
    
    # Get upsampling ratios and sort
    upsampling_ratios = sorted(df_filtered['llm_upsampling_ratio'].unique())
    
    # Create heatmap data
    heatmap_data = []
    probe_labels = []
    
    # Debug: Check what probe names are actually in the data
    available_probes = df_filtered['probe_name'].unique()
    # Filter out None values before sorting
    available_probes_clean = [p for p in available_probes if p is not None]
    print(f"Debug - Available probe names in LLM upsampling data: {sorted(available_probes_clean)}")
    print(f"Debug - Looking for attention probes: {[p for p in probe_names if 'attention' in p]}")
    
    for probe in probe_names:
        probe_data = df_filtered[df_filtered['probe_name'] == probe]
        print(f"Debug - Probe '{probe}': {len(probe_data)} rows found")
        if not probe_data.empty:
            # Average across all positive samples (1-10) for each upsampling ratio
            ratio_means = []
            for ratio in upsampling_ratios:
                ratio_probe_data = probe_data[probe_data['llm_upsampling_ratio'] == ratio]
                if not ratio_probe_data.empty:
                    mean_value = ratio_probe_data[metric].mean()
                    ratio_means.append(mean_value)
                else:
                    ratio_means.append(np.nan)
            
            heatmap_data.append(ratio_means)
            probe_labels.append(get_detailed_probe_label(probe))
        else:
            print(f"Debug - No data found for probe '{probe}'")
    
    if not heatmap_data:
        print(f"No data to plot for heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap with smaller cells for 16 probes
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create heatmap
    im = ax.imshow(heatmap_array, cmap='viridis', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(upsampling_ratios)))
    ax.set_xticklabels([f'{ratio}x' for ratio in upsampling_ratios])
    ax.set_yticks(range(len(probe_labels)))
    ax.set_yticklabels(probe_labels)
    
    # Add x-axis label
    ax.set_xlabel('Upsampling Factor', fontsize=16, labelpad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if metric == 'recall':
        cbar.set_label('Recall @ FPR=0.01 (Averaged Over 1-10 Samples)', rotation=270, labelpad=20)
    else:
        cbar.set_label(metric.upper() + ' (Averaged Over 1-10 Samples)', rotation=270, labelpad=20)
    
    # Add text annotations with dynamic coloring
    for i in range(len(probe_labels)):
        for j in range(len(upsampling_ratios)):
            if not np.isnan(heatmap_array[i, j]):
                # Determine text color based on background brightness
                value = heatmap_array[i, j]
                # Use white text for dark backgrounds, black for light backgrounds
                text_color = 'white' if value < 0.8 else 'black'
                text = ax.text(j, i, f'{value:.2f}', 
                             ha='center', va='center', color=text_color, fontsize=14)
    
    # Set title with better description and line breaks
    title_suffix = '(AUC)' if metric == 'auc' else '(Recall @ FPR=0.01)'
    if '87_is_spam' in eval_dataset:
        title = f'OOD Generalization: LLM-Upsampled Gemma-2-9b Probes\nEvaluated On SMS Spam {title_suffix}'
    else:
        if 'Recall' in title_suffix:
            title = f'LLM-Upsampled Gemma-2-9b Probes On Enron-Spam\n{title_suffix}'
        else:
            title = f'LLM-Upsampled Gemma-2-9b Probes On Enron-Spam {title_suffix}'
    ax.set_title(title, fontsize=20, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# This function is now imported from viz_util.py


if __name__ == "__main__":
    generate_all_visualizations()
