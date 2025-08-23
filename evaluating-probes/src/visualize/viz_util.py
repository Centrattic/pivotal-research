"""
Visualization Utilities

This module contains reusable visualization methods that can be used by both
plot_generator.py (for default probes) and hyperparameter_analysis.py (for best probes).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import seaborn as sns
from scipy import stats
import re
import matplotlib.ticker as mticker

try:
    from .data_loader import (
        get_data_for_visualization,
        get_probe_names,
        get_eval_datasets,
        get_run_names,
        load_metrics_data,
        extract_info_from_filename,
        filter_gemma_sae_topk_1024,
        filter_linear_probes_c_1_0,
        filter_default_attention_probes
    )
except ImportError:
    from data_loader import (
        get_data_for_visualization,
        get_probe_names,
        get_eval_datasets,
        get_run_names,
        load_metrics_data,
        extract_info_from_filename,
        filter_gemma_sae_topk_1024,
        filter_linear_probes_c_1_0,
        filter_default_attention_probes
    )


def setup_plot_style(figsize=(8, 6), big_figsize=(16, 12)):
    """Set up consistent plot styling."""
    plt.style.use('default')
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['legend.title_fontsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 8
    return figsize, big_figsize


def get_probe_label(probe_name: str) -> str:
    """Get human-readable label for probe names."""
    label_map = {
        'act_sim_last': 'Cosine Similarity (last)',
        'act_sim_max': 'Cosine Similarity (max)',
        'act_sim_mean': 'Cosine Similarity (mean)',
        'act_sim_softmax': 'Cosine Similarity (softmax)',
        'sae_last': 'SAE (last)',
        'sae_max': 'SAE (max)',
        'sae_mean': 'SAE (mean)',
        'sae_softmax': 'SAE (softmax)',
        'sklearn_linear_last': 'Linear (last)',
        'sklearn_linear_max': 'Linear (max)',
        'sklearn_linear_mean': 'Linear (mean)',
        'sklearn_linear_softmax': 'Linear (softmax)',
        'attention': 'Attention',
    }
    return label_map.get(probe_name, probe_name)


def get_detailed_probe_label(probe_name: str, run_name: str = None) -> str:
    """Get detailed probe label with consistent formatting."""
    # Map probe names to readable labels
    probe_labels = {
        'act_sim_last': 'Cosine Similarity (last)',
        'act_sim_max': 'Cosine Similarity (max)',
        'act_sim_mean': 'Cosine Similarity (mean)',
        'act_sim_softmax': 'Cosine Similarity (softmax)',
        'sklearn_linear_last': 'Linear (last)',
        'sklearn_linear_max': 'Linear (max)',
        'sklearn_linear_mean': 'Linear (mean)',
        'sklearn_linear_softmax': 'Linear (softmax)',
        'sae_last': 'SAE (last)',
        'sae_max': 'SAE (max)',
        'sae_mean': 'SAE (mean)',
        'sae_softmax': 'SAE (softmax)',
        'attention_last': 'Attention (last)',
        'attention_max': 'Attention (max)',
        'attention_mean': 'Attention (mean)',
        'attention_softmax': 'Attention (softmax)'
    }
    
    return probe_labels.get(probe_name, probe_name)


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


def add_clean_log_inset(ax, pos=(0.58, 0.12, 0.35, 0.35), *,
                        xlim=None, ylim=(0.97, 1.00), labelsize=12):
    """
    Create a 'clean' inset on ax with a log x-axis and only 10^n labels.
    pos is [left, bottom, width, height] in axes coordinates.
    """
    # Use Matplotlib's built-in inset creator (handles layering better)
    axin = ax.inset_axes(pos, zorder=10)

    # Make it opaque so nothing shows through
    axin.set_facecolor("white")
    for s in axin.spines.values():
        s.set_linewidth(1)

    if xlim is not None:
        axin.set_xlim(*xlim)
    if ylim is not None:
        axin.set_ylim(*ylim)

    # Log x with only 10^n major labels; hide all minor labels
    axin.set_xscale('log')
    axin.xaxis.set_major_locator(mticker.LogLocator(base=10))
    axin.xaxis.set_major_formatter(mticker.LogFormatter(base=10, labelOnlyBase=True))
    # NOTE: subs must be FRACTIONS of a decade for minors:
    axin.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
    axin.xaxis.set_minor_formatter(mticker.NullFormatter())
    axin.tick_params(which='minor', labelbottom=False)  # belt & suspenders
    axin.tick_params(labelsize=labelsize, pad=1)
    axin.grid(False)

    # Ensure parent ticklabels are *behind* the inset
    for t in ax.get_xticklabels() + ax.get_xticklabels(minor=True):
        t.set_zorder(1)
    return axin


def get_best_probes_by_category(df: pd.DataFrame) -> List[str]:
    """Get the best probe from each category (attention, SAE, linear, act_sim)."""
    # Group probes by category
    probe_categories = {
        'attention': [],
        'sae': [],
        'sklearn_linear': [],
        'act_sim': []
    }
    
    for probe_name in df['probe_name'].unique():
        if probe_name and probe_name != 'attention':
            if 'sae' in probe_name:
                probe_categories['sae'].append(probe_name)
            elif 'sklearn_linear' in probe_name:
                probe_categories['sklearn_linear'].append(probe_name)
            elif 'act_sim' in probe_name:
                probe_categories['act_sim'].append(probe_name)
    
    # Always include attention if available
    best_probes = []
    if 'attention' in df['probe_name'].values:
        best_probes.append('attention')
    
    # Find best probe from each category
    for category, probes in probe_categories.items():
        if probes:
            # Calculate median AUC for each probe in this category
            probe_scores = []
            for probe in probes:
                probe_data = df[df['probe_name'] == probe]
                if not probe_data.empty:
                    median_auc = probe_data['auc'].median()
                    probe_scores.append((probe, median_auc))
            
            if probe_scores:
                # Sort by median AUC and take the best
                best_probe = max(probe_scores, key=lambda x: x[1])[0]
                best_probes.append(best_probe)
    
    return best_probes[:4]  # Return top 4


def apply_main_plot_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply filters for main plots:
    - For Gemma SAEs: only topk=1024 (exclude 262k SAEs)
    - For linear probes: only C=1.0 (or no C specified)
    - For attention probes: only default probes (no wd_ and lr_ parameters)
    """
    # Debug: entry counts
    try:
        print(f"[filters] start: rows={len(df)} | types={df['probe_name'].value_counts().to_dict()}")
    except Exception:
        print(f"[filters] start: rows={len(df)}")

    # Only target Gemma SAE rows; keep everything else untouched
    is_gemma = df['run_name'].str.contains('gemma', case=False, na=False)
    is_sae = df['probe_name'].str.contains('sae', na=False)
    gemma_sae_mask = is_gemma & is_sae
    print(f"[filters] gemma_sae_mask true rows={gemma_sae_mask.sum()} of {len(df)}")

    gemma_sae_df = df[gemma_sae_mask]
    non_gemma_or_non_sae_df = df[~gemma_sae_mask]

    # Apply Gemma SAE topk=1024 filter only to gemma+sae rows
    filtered_gemma_sae_df = filter_gemma_sae_topk_1024(gemma_sae_df)
    print(f"[filters] gemma+sae after topk=1024 filter: rows={len(filtered_gemma_sae_df)}")

    # Merge back
    combined_df = pd.concat([filtered_gemma_sae_df, non_gemma_or_non_sae_df], ignore_index=True)
    print(f"[filters] combined after gemma-sae step: rows={len(combined_df)}")
    
    # Split by probe type for specific filtering
    linear_probes = combined_df[combined_df['probe_name'].str.contains('sklearn_linear', na=False)]
    attention_probes = combined_df[combined_df['probe_name'].str.contains('attention', na=False)]
    other_probes = combined_df[
        (~combined_df['probe_name'].str.contains('sklearn_linear', na=False)) & 
        (~combined_df['probe_name'].str.contains('attention', na=False))
    ]
    print(f"[filters] split types: linear={len(linear_probes)}, attention={len(attention_probes)}, other={len(other_probes)}")
    
    # Apply specific filters
    filtered_linear_probes = filter_linear_probes_c_1_0(linear_probes)
    filtered_attention_probes = filter_default_attention_probes(attention_probes)
    print(f"[filters] after specific: linear={len(filtered_linear_probes)}, attention={len(filtered_attention_probes)}")
    
    # Combine all filtered data
    filtered_df = pd.concat([filtered_linear_probes, filtered_attention_probes, other_probes], ignore_index=True)
    print(f"[filters] final combined rows={len(filtered_df)} | types={filtered_df['probe_name'].value_counts().to_dict() if not filtered_df.empty else {}}")

    # Safety fallback to avoid empty plots while debugging
    if filtered_df.empty:
        print("[filters] WARNING: filtering produced 0 rows. Returning unfiltered data temporarily.")
        return df

    return filtered_df


def calculate_confidence_interval(data: List[float], confidence: float = 0.9) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    # Filter out NaN and inf values
    clean_data = [x for x in data if not (np.isnan(x) or np.isinf(x))]
    
    if len(clean_data) < 2:
        if len(clean_data) == 1:
            return clean_data[0], clean_data[0]
        else:
            return 0.0, 0.0
    
    mean_val = np.mean(clean_data)
    std_err = stats.sem(clean_data)
    
    # Check for valid standard error
    if np.isnan(std_err) or np.isinf(std_err) or std_err == 0:
        return mean_val, mean_val
    
    try:
        confidence_interval = stats.t.interval(confidence, len(clean_data) - 1, loc=mean_val, scale=std_err)
        lower, upper = confidence_interval[0], confidence_interval[1]
        
        # Check for valid bounds
        if np.isnan(lower) or np.isinf(lower):
            lower = mean_val
        if np.isnan(upper) or np.isinf(upper):
            upper = mean_val
            
        return lower, upper
    except:
        # Fallback to mean if confidence interval calculation fails
        return mean_val, mean_val


def format_run_name_for_display(run_name: str) -> str:
    """Format run_name for display in plot titles."""
    if 'gemma' in run_name.lower():
        # Extract model size and format as "Gemma-2-9b"
        if '9b' in run_name.lower():
            return "Gemma-2-9b"
        elif '7b' in run_name.lower():
            return "Gemma-2-7b"
        else:
            return "Gemma-2"
    elif 'qwen' in run_name.lower():
        # Extract model size and format as "Qwen-1.5-7B"
        if '32b' in run_name.lower():
            return "Qwen-3-32B"
        elif '14b' in run_name.lower():
            return "Qwen-3-14B"
        elif '8b' in run_name.lower():
            return "Qwen-3-8B"
        elif '4b' in run_name.lower():
            return "Qwen-3-4B"
        elif '1.7b' in run_name.lower():
            return "Qwen-3-1.7B"
        elif '0.6b' in run_name.lower():
            return "Qwen-3-0.6B"
        else:
            return "Qwen-1.5"
    elif 'mask' in run_name.lower():
        return "LLama-3.3-70b"
    else:
        return run_name


def format_dataset_name_for_display(eval_dataset: str) -> str:
    """Format eval_dataset for display in plot titles."""
    if 'enron' in eval_dataset.lower() or 'spam' in eval_dataset.lower():
        return "Enron-Spam"
    elif 'mask' in eval_dataset.lower():
        return "MASK"
    else:
        # Capitalize first letter and replace underscores with spaces
        return eval_dataset.replace('_', ' ').title()


def extract_model_size(run_name: str) -> int:
    """Extract model size from run name for sorting."""
    # Extract size from names like "spam_gemma_2b", "spam_qwen_0.6B", etc.
    if 'gemma' in run_name.lower():
        if '2b' in run_name.lower():
            return 2
        elif '9b' in run_name.lower():
            return 9
        elif '27b' in run_name.lower():
            return 27
    elif 'qwen' in run_name.lower():
        if '0.6b' in run_name.lower():
            return 1  # 0.6B
        elif '1.7b' in run_name.lower():
            return 2  # 1.7B
        elif '14b' in run_name.lower():
            return 14  # Check 14B before 4B to avoid partial match
        elif '4b' in run_name.lower():
            return 4
        elif '8b' in run_name.lower():
            return 8
        elif '32b' in run_name.lower():
            return 32
    elif 'mask' in run_name.lower():
        if '70b' in run_name.lower():
            return 70
    return 0  # Default for unknown sizes


def get_default_hyperparameter_filters() -> Dict[str, Any]:
    """
    Get the default hyperparameter filters used in the original plot_generator.
    
    Returns:
        Dictionary mapping probe types to their default hyperparameter filters
    """
    return {
        'sae': {
            'topk': 1024,  # Only 1024, exclude 262k SAEs
            'exclude_262k': True
        },
        'sklearn_linear': {
            'C': 1.0  # Only C=1.0 or no C specified
        },
        'attention': {
            'default_only': True  # Only default probes (no wd_ and lr_ parameters)
        }
    }


def filter_data_by_hyperparameters(
    df: pd.DataFrame,
    probe_type: str,
    hyperparams: Dict[str, Any]
) -> pd.DataFrame:
    """
    Filter data by hyperparameters for a specific probe type.
    
    Args:
        df: DataFrame to filter
        probe_type: Type of probe ('sae', 'sklearn_linear', 'attention')
        hyperparams: Dictionary of hyperparameter filters
        
    Returns:
        Filtered DataFrame
    """
    if probe_type == 'sae':
        # Filter for SAE probes
        mask = df['probe_name'].str.contains('sae', na=False)
        
        # Apply topk filter if specified
        if 'topk' in hyperparams:
            mask &= (df['topk'] == hyperparams['topk'])
        
        # Exclude 262k SAEs if requested
        if hyperparams.get('exclude_262k', False):
            mask &= (~df['filename'].str.contains('sae_262k', na=False))
            mask &= (~df['filename'].str.contains('sae2_', na=False))
            mask &= (~df['filename'].str.contains('262k', na=False))
        
        return df[mask]
    
    elif probe_type == 'sklearn_linear':
        # Filter for linear probes
        mask = df['probe_name'].str.contains('sklearn_linear', na=False)
        
        # Apply C filter if specified
        if 'C' in hyperparams:
            mask &= (
                (df['C'] == hyperparams['C']) | 
                (df['C'].isna()) |  # No C specified, defaults to 1.0
                (df['filename'].str.contains('C1e0', na=False))  # C=1e0 format
            )
        
        return df[mask]
    
    elif probe_type == 'attention':
        # Filter for attention probes
        mask = df['probe_name'].str.contains('attention', na=False)
        
        # Apply default-only filter if specified
        if hyperparams.get('default_only', False):
            mask &= (
                (
                    (~df['filename'].str.contains('wd_', na=False)) &  # No weight decay parameter
                    (~df['filename'].str.contains('lr_', na=False))    # No learning rate parameter
                ) |
                (df['filename'].str.contains('lr_6p31e-04_wd_0p00e00', na=False))  # Specific default values
            )
        else:
            # Apply specific hyperparameter filters
            if 'lr' in hyperparams:
                mask &= (df['lr'] == hyperparams['lr'])
            if 'weight_decay' in hyperparams:
                mask &= (df['weight_decay'] == hyperparams['weight_decay'])
        
        return df[mask]
    
    else:
        # For other probe types, return unfiltered
        return df


def get_best_probes_by_category_with_hyperparams(
    df: pd.DataFrame,
    hyperparam_filters: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[str]:
    """
    Get the best probe from each category with optional hyperparameter filtering.
    
    Args:
        df: DataFrame with probe data
        hyperparam_filters: Dictionary mapping probe types to hyperparameter filters
        
    Returns:
        List of best probe names
    """
    if hyperparam_filters is None:
        hyperparam_filters = get_default_hyperparameter_filters()
    
    # Group probes by category
    probe_categories = {
        'attention': [],
        'sae': [],
        'sklearn_linear': [],
        'act_sim': []
    }
    
    for probe_name in df['probe_name'].unique():
        if probe_name and probe_name != 'attention':
            if 'sae' in probe_name:
                probe_categories['sae'].append(probe_name)
            elif 'sklearn_linear' in probe_name:
                probe_categories['sklearn_linear'].append(probe_name)
            elif 'act_sim' in probe_name:
                probe_categories['act_sim'].append(probe_name)
    
    # Always include attention if available
    best_probes = []
    if 'attention' in df['probe_name'].values:
        best_probes.append('attention')
    
    # Find best probe from each category
    for category, probes in probe_categories.items():
        if probes:
            # Filter by hyperparameters if specified
            if category in hyperparam_filters:
                category_df = filter_data_by_hyperparameters(df, category, hyperparam_filters[category])
            else:
                category_df = df[df['probe_name'].isin(probes)]
            
            if not category_df.empty:
                # Calculate median AUC for each probe in this category
                probe_scores = []
                for probe in probes:
                    probe_data = category_df[category_df['probe_name'] == probe]
                    if not probe_data.empty:
                        median_auc = probe_data['auc'].median()
                        probe_scores.append((probe, median_auc))
                
                if probe_scores:
                    # Sort by median AUC and take the best
                    best_probe = max(probe_scores, key=lambda x: x[1])[0]
                    best_probes.append(best_probe)
    
    return best_probes[:4]  # Return top 4


def plot_experiment_best_probes_generic(
    eval_dataset: str,
    run_name: str,
    save_path: Path,
    experiment: str,
    metric: str = 'auc',
    hyperparam_filters: Optional[Dict[str, Dict[str, Any]]] = None,
    title_suffix: str = "",
    output_dir: Optional[Path] = None
):
    """
    Generic function to plot best probes for any experiment with custom hyperparameter filters.
    
    Args:
        eval_dataset: Evaluation dataset name
        run_name: Model run name
        save_path: Path to save the plot
        experiment: Experiment type ('2-', '4-', etc.)
        metric: Metric to plot ('auc' or 'recall')
        hyperparam_filters: Hyperparameter filters for each probe type
        title_suffix: Additional text for the title
        output_dir: Optional output directory to override save_path
    """
    plot_size, _ = setup_plot_style()
    
    # Get data for the experiment
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        exclude_attention=False,
        include_val_eval=False  # Always use test_eval for plotting
    )
    
    if df.empty:
        print(f"No data found for experiment {experiment}, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Apply hyperparameter filters if specified, otherwise use default filters
    if hyperparam_filters is not None:
        # Use custom hyperparameter filtering
        best_probes = get_best_probes_by_category_with_hyperparams(df, hyperparam_filters)
    else:
        # Use default filtering (same as plot_generator)
        df = apply_main_plot_filters(df)
        best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment {experiment}, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_size)
    
    # Define consistent colors for each probe category
    color_map = {
        'attention': 'red',
        'sae': 'orange', 
        'sklearn_linear': 'green',
        'act_sim': 'blue'
    }
    
    all_y_values = []
    lines_plotted = 0
    
    for probe in best_probes:
        probe_data = df[df['probe_name'] == probe]
        
        if probe_data.empty:
            continue
        
        # Group by number of positive samples
        grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
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
        
        # Add small x-jitter to prevent overlapping points
        x_jitter = np.random.normal(0, 0.02, len(x_values))
        x_values_jittered = x_values * (1 + x_jitter)
        
        # Determine color based on probe category
        color = 'gray'  # default
        if 'attention' in probe:
            color = color_map['attention']
        elif 'sae' in probe:
            color = color_map['sae']
        elif 'sklearn_linear' in probe:
            color = color_map['sklearn_linear']
        elif 'act_sim' in probe:
            color = color_map['act_sim']
        
        # Plot line with confidence interval
        ax.plot(x_values_jittered, y_means, 'o-', color=color, label=get_probe_label(probe), 
                linewidth=4, markersize=8)
        ax.fill_between(x_values_jittered, y_lower, y_upper, alpha=0.3, color=color)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for experiment {experiment} {metric}, eval_dataset={eval_dataset}, run_name={run_name}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # Get number of negative examples for title
    num_negative = df['num_negative_samples'].iloc[0] if not df.empty and 'num_negative_samples' in df.columns else None
    
    # Format run_name for cleaner title
    formatted_run_name = format_run_name_for_display(run_name)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'AUC'
    elif metric == 'recall':
        ylabel = 'Recall @ FPR=0.01'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    
    # Create title based on experiment type
    if experiment == '2-':
        if num_negative is not None:
            num_negative_int = int(num_negative)
            title = f'{formatted_run_name} Probes On Unbalanced Enron-Spam Dataset\n({num_negative_int} negative examples){title_suffix}'
        else:
            title = f'{formatted_run_name} Probes On Unbalanced Enron-Spam Dataset{title_suffix}'
    elif experiment == '4-':
        title = f'{formatted_run_name} Probes on Balanced Enron-Spam Dataset{title_suffix}'
    else:
        title = f'{formatted_run_name} Probes - {experiment} {title_suffix}'
    
    ax.set_title(title, y=1.02)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=-0.5)
    
    # Adjust subplot parameters to reduce gap
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
    
    # Ensure output directory exists
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / save_path.name
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probe_subplots_generic(
    experiment: str,
    eval_dataset: str,
    run_name: str,
    save_path: Path,
    metric: str = 'auc',
    hyperparam_filters: Optional[Dict[str, Dict[str, Any]]] = None,
    title_suffix: str = "",
    output_dir: Optional[Path] = None
):
    """
    Generic function to create subplot visualization for all probes with custom hyperparameter filters.
    
    Args:
        experiment: Experiment type
        eval_dataset: Evaluation dataset name
        run_name: Model run name
        save_path: Path to save the plot
        metric: Metric to plot ('auc' or 'recall')
        hyperparam_filters: Hyperparameter filters for each probe type
        title_suffix: Additional text for the title
        output_dir: Optional output directory to override save_path
    """
    _, plot_size = setup_plot_style()
    
    # Get data for the experiment
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        exclude_attention=True,
        include_val_eval=False  # Always use test_eval for plotting
    )
    
    if df.empty:
        print(f"No data found for experiment {experiment}, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Apply hyperparameter filters if specified
    if hyperparam_filters:
        filtered_dfs = []
        for probe_type, filters in hyperparam_filters.items():
            filtered_df = filter_data_by_hyperparameters(df, probe_type, filters)
            filtered_dfs.append(filtered_df)
        df = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else df
    
    # Define the probe names we want to include
    if 'gemma' in run_name.lower():
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    else:
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    
    # Create subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    # Define colors for each probe type
    act_sim_color = 'blue'
    sae_color = 'orange'
    linear_color = 'green'
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'AUC'
        title_suffix_plot = '(AUC)'
    elif metric == 'recall':
        ylabel = 'Recall @ FPR=0.01'
        title_suffix_plot = '(Recall @ FPR=0.01)'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    all_y_values = []
    
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
            
            # Calculate confidence intervals for shaded regions
            y_lower = []
            y_upper = []
            for values in grouped[metric]:
                lower, upper = calculate_confidence_interval(values)
                y_lower.append(lower)
                y_upper.append(upper)
            
            # Plot line with confidence interval
            ax.plot(x_values, y_means, 'o-', color=color, markersize=6, linewidth=2)
            ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=color)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
            ax.set_xlabel('Num Positive Samples')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25), y=1.05)
    
    # Set dynamic ylim for each subplot based on its own data
    for i, probe in enumerate(probe_names):
        ax = axes[i]
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty and probe_data['num_positive_samples'].notna().any():
            grouped = probe_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
            grouped = grouped.sort_values('num_positive_samples')
            
            if not grouped.empty:
                probe_y_values = []
                for values in grouped[metric]:
                    probe_y_values.extend(values)
                
                if probe_y_values:
                    probe_y_min = max(0.70, min(probe_y_values) - 0.05)
                    probe_y_max = min(1.0, max(probe_y_values) + 0.05)
                    if probe_y_min < probe_y_max:
                        ax.set_ylim(probe_y_min, probe_y_max)
                    else:
                        ax.set_ylim(0.0, 1.0)
    
    # Format run name for display
    formatted_run_name = format_run_name_for_display(run_name)
    
    # Create title
    if experiment == '2-':
        if '87_is_spam' in eval_dataset:
            title = f"{formatted_run_name} Probes Trained on Unbalanced Enron-Spam Dataset and Evaluated on OOD SMS Spam Dataset{title_suffix}"
        else:
            title = f"{formatted_run_name} Probes Trained and Evaluated on Unbalanced Enron-Spam Dataset (1750 negative examples){title_suffix}"
    elif experiment == '4-':
        title = f"{formatted_run_name} Probes Trained and Evaluated on Balanced Enron-Spam Dataset{title_suffix}"
    else:
        title = f"{formatted_run_name} Probes - {experiment} {title_suffix_plot} {eval_dataset}{title_suffix}"
    
    fig.suptitle(title, fontsize=24, y=0.98)
    
    # Ensure output directory exists
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / save_path.name
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
