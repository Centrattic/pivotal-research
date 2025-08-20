import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from scipy import stats
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
        'attention': 'Attention',
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


def plot_experiment_2_best_probes_auc(eval_dataset: str, save_path: Path):
    """Plot experiment 2 best probes AUC as line chart with confidence intervals."""
    setup_plot_style()
    
    # Get data for experiment 2
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        exclude_attention=False  # Include attention for best probes
    )
    
    if df.empty:
        print(f"No data found for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    all_y_values = []  # Collect all y values for proper ylim
    lines_plotted = 0
    
    for i, probe in enumerate(best_probes):
        probe_data = df[df['probe_name'] == probe]
        
        if probe_data.empty:
            continue
        
        # Group by number of positive samples
        grouped = probe_data.groupby('num_positive_samples')['auc'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_means = [np.mean(auc_list) for auc_list in grouped['auc']]
        y_lower = []
        y_upper = []
        
        for auc_list in grouped['auc']:
            lower, upper = calculate_confidence_interval(auc_list)
            y_lower.append(lower)
            y_upper.append(upper)
        
        # Collect all y values for ylim calculation
        all_y_values.extend(y_means)
        all_y_values.extend(y_lower)
        all_y_values.extend(y_upper)
        
        # Plot line with confidence interval
        ax.plot(x_values, y_means, 'o-', color=colors[i], label=get_probe_label(probe), 
                linewidth=2, markersize=4)
        ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=colors[i])
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for experiment 2 AUC, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('AUC')
    ax.set_title(f'Experiment 2: Best Probes AUC Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_2_best_probes_recall(eval_dataset: str, save_path: Path):
    """Plot experiment 2 best probes Recall @ FPR=0.01 as line chart with confidence intervals."""
    setup_plot_style()
    
    # Get data for experiment 2
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        exclude_attention=False  # Include attention for best probes
    )
    
    if df.empty:
        print(f"No data found for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(best_probes):
        probe_data = df[df['probe_name'] == probe]
        
        if probe_data.empty:
            continue
        
        # Group by number of positive samples
        grouped = probe_data.groupby('num_positive_samples')['recall'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        x_values = grouped['num_positive_samples'].values
        y_means = [np.mean(recall_list) for recall_list in grouped['recall']]
        y_lower = []
        y_upper = []
        
        for recall_list in grouped['recall']:
            lower, upper = calculate_confidence_interval(recall_list)
            y_lower.append(lower)
            y_upper.append(upper)
        
        # Collect all y values for ylim calculation
        all_y_values.extend(y_means)
        all_y_values.extend(y_lower)
        all_y_values.extend(y_upper)
        
        # Plot line with confidence interval
        ax.plot(x_values, y_means, 'o-', color=colors[i], label=get_probe_label(probe), 
                linewidth=2, markersize=4)
        ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=colors[i])
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Recall @ FPR=0.01')
    ax.set_title(f'Experiment 2: Best Probes Recall Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_4_best_probes_auc(eval_dataset: str, save_path: Path):
    """Plot experiment 4 best probes AUC as line chart with confidence intervals."""
    setup_plot_style()
    
    # Get data for experiment 4
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='4-',
        exclude_attention=False  # Include attention for best probes
    )
    
    if df.empty:
        print(f"No data found for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(best_probes):
        probe_data = df[df['probe_name'] == probe]
        
        if probe_data.empty:
            continue
        
        # Group by number of positive samples
        grouped = probe_data.groupby('num_positive_samples')['auc'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        x_values = grouped['num_positive_samples'].values
        y_means = [np.mean(auc_list) for auc_list in grouped['auc']]
        y_lower = []
        y_upper = []
        
        for auc_list in grouped['auc']:
            lower, upper = calculate_confidence_interval(auc_list)
            y_lower.append(lower)
            y_upper.append(upper)
        
        # Collect all y values for ylim calculation
        all_y_values.extend(y_means)
        all_y_values.extend(y_lower)
        all_y_values.extend(y_upper)
        
        # Plot line with confidence interval
        ax.plot(x_values, y_means, 'o-', color=colors[i], label=get_probe_label(probe), 
                linewidth=2, markersize=4)
        ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=colors[i])
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('AUC')
    ax.set_title(f'Experiment 4: Best Probes AUC Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_4_best_probes_recall(eval_dataset: str, save_path: Path):
    """Plot experiment 4 best probes Recall @ FPR=0.01 as line chart with confidence intervals."""
    setup_plot_style()
    
    # Get data for experiment 4
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='4-',
        exclude_attention=False  # Include attention for best probes
    )
    
    if df.empty:
        print(f"No data found for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(best_probes):
        probe_data = df[df['probe_name'] == probe]
        
        if probe_data.empty:
            continue
        
        # Group by number of positive samples
        grouped = probe_data.groupby('num_positive_samples')['recall'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        x_values = grouped['num_positive_samples'].values
        y_means = [np.mean(recall_list) for recall_list in grouped['recall']]
        y_lower = []
        y_upper = []
        
        for recall_list in grouped['recall']:
            lower, upper = calculate_confidence_interval(recall_list)
            y_lower.append(lower)
            y_upper.append(upper)
        
        # Collect all y values for ylim calculation
        all_y_values.extend(y_means)
        all_y_values.extend(y_lower)
        all_y_values.extend(y_upper)
        
        # Plot line with confidence interval
        ax.plot(x_values, y_means, 'o-', color=colors[i], label=get_probe_label(probe), 
                linewidth=2, markersize=4)
        ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=colors[i])
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Recall @ FPR=0.01')
    ax.set_title(f'Experiment 4: Best Probes Recall Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_aggregated_auc(eval_dataset: str, save_path: Path):
    """Plot LLM upsampling with median AUC performance across all probes."""
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
    
    # Get unique upsampling ratios
    upsampling_ratios = sorted(df['llm_upsampling_ratio'].dropna().unique())
    
    if not upsampling_ratios:
        print(f"No upsampling ratios found for eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
    all_y_values = []
    lines_plotted = 0
    
    for i, ratio in enumerate(upsampling_ratios):
        ratio_data = df[df['llm_upsampling_ratio'] == ratio]
        
        if ratio_data.empty:
            continue
        
        # Group by number of positive samples and calculate median across all probes
        grouped = ratio_data.groupby('num_positive_samples')['auc'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(auc_list) for auc_list in grouped['auc']]
        all_y_values.extend(y_medians)
        
        # Plot line
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{ratio}x upsampling', linewidth=2, markersize=4)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for LLM upsampling AUC, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Median AUC (across all probes)')
    ax.set_title(f'LLM Upsampling: Median AUC Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_aggregated_recall(eval_dataset: str, save_path: Path):
    """Plot LLM upsampling with median Recall performance across all probes."""
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
    
    # Get unique upsampling ratios
    upsampling_ratios = sorted(df['llm_upsampling_ratio'].dropna().unique())
    
    if not upsampling_ratios:
        print(f"No upsampling ratios found for eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
    all_y_values = []
    lines_plotted = 0
    
    for i, ratio in enumerate(upsampling_ratios):
        ratio_data = df[df['llm_upsampling_ratio'] == ratio]
        
        if ratio_data.empty:
            continue
        
        # Group by number of positive samples and calculate median across all probes
        grouped = ratio_data.groupby('num_positive_samples')['recall'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(recall_list) for recall_list in grouped['recall']]
        all_y_values.extend(y_medians)
        
        # Plot line
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{ratio}x upsampling', linewidth=2, markersize=4)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for LLM upsampling Recall, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Median Recall @ FPR=0.01 (across all probes)')
    ax.set_title(f'LLM Upsampling: Median Recall Performance\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_aggregated_auc(eval_dataset: str, save_path: Path):
    """Plot scaling law with median AUC performance across all probes."""
    setup_plot_style()
    
    # Get Qwen run names for scaling analysis
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    qwen_runs.sort(key=lambda x: float(x.split('_')[-1].replace('b', '').replace('B', '')))
    
    if not qwen_runs:
        print(f"No Qwen runs found for scaling law, eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
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
            continue
        
        # Extract model size for legend
        size_str = run.split('_')[-1].replace('b', '').replace('B', '')
        model_size = float(size_str)
        
        # Group by number of positive samples and calculate median across all probes
        grouped = df.groupby('num_positive_samples')['auc'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(auc_list) for auc_list in grouped['auc']]
        all_y_values.extend(y_medians)
        
        # Plot line
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{model_size}B', linewidth=2, markersize=4)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for scaling law AUC, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Median AUC (across all probes)')
    ax.set_title(f'Scaling across Qwen model sizes: AUC\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_aggregated_recall(eval_dataset: str, save_path: Path):
    """Plot scaling law with median Recall performance across all probes."""
    setup_plot_style()
    
    # Get Qwen run names for scaling analysis
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    qwen_runs.sort(key=lambda x: float(x.split('_')[-1].replace('b', '').replace('B', '')))
    
    if not qwen_runs:
        print(f"No Qwen runs found for scaling law, eval_dataset={eval_dataset}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
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
            continue
        
        # Extract model size for legend
        size_str = run.split('_')[-1].replace('b', '').replace('B', '')
        model_size = float(size_str)
        
        # Group by number of positive samples and calculate median across all probes
        grouped = df.groupby('num_positive_samples')['recall'].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(recall_list) for recall_list in grouped['recall']]
        all_y_values.extend(y_medians)
        
        # Plot line
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{model_size}B', linewidth=2, markersize=4)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for scaling law Recall, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Median Recall @ FPR=0.01 (across all probes)')
    ax.set_title(f'Scaling across Qwen model sizes: Recall\nEval Dataset: {wrap_text(eval_dataset)}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probe_subplots(experiment: str, eval_dataset: str, save_path: Path):
    """Create subplot visualization for all probes (3x4 grid, excluding attention)."""
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
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()
    
    # Define colors for each probe type
    act_sim_color = 'blue'
    sae_color = 'orange'
    linear_color = 'green'
    
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(probe_names[:12]):  # Limit to 12 probes
        ax = axes[i]
        
        # Get data for this probe
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty and probe_data['num_positive_samples'].notna().any():
            # Group by number of positive samples
            grouped = probe_data.groupby('num_positive_samples')['auc'].apply(list).reset_index()
            grouped = grouped.sort_values('num_positive_samples')
            
            x_values = grouped['num_positive_samples'].values
            y_means = [np.mean(auc_list) for auc_list in grouped['auc']]
            all_y_values.extend(y_means)
            
            # Choose color based on probe type
            if 'act_sim' in probe:
                color = act_sim_color
            elif 'sae' in probe:
                color = sae_color
            elif 'sklearn_linear' in probe:
                color = linear_color
            else:
                color = 'black'
            
            ax.plot(x_values, y_means, 'o-', color=color, markersize=4, linewidth=1.5)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Training Size')
            ax.set_ylabel('AUC')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    
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
            # Group by upsampling factor and number of positive samples
            grouped = probe_data.groupby(['llm_upsampling_ratio', 'num_positive_samples'])['auc'].apply(list).reset_index()
            
            # Get unique upsampling ratios
            upsampling_ratios = sorted(probe_data['llm_upsampling_ratio'].dropna().unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
            
            for j, ratio in enumerate(upsampling_ratios):
                ratio_data = grouped[grouped['llm_upsampling_ratio'] == ratio]
                if not ratio_data.empty:
                    ratio_data = ratio_data.sort_values('num_positive_samples')
                    x_values = ratio_data['num_positive_samples'].values
                    y_medians = [np.median(auc_list) for auc_list in ratio_data['auc']]
                    
                    ax.plot(x_values, y_medians, 'o-', color=colors[j], 
                           label=f'{ratio}x', markersize=3, linewidth=1)
            
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Positive Samples')
            ax.set_ylabel('Median AUC')
            ax.set_ylim(0.6, 1.0)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            if i == 0:  # Only show legend on first subplot
                ax.legend(title='Upsampling', fontsize=8)
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
            ax.set_ylim(0.6, 1.0)
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


def plot_scaling_law_heatmap_auc(eval_dataset: str, save_path: Path):
    """Plot scaling law heatmap for AUC with probes vs model sizes."""
    setup_plot_style()
    
    # Get Qwen run names for scaling analysis
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    qwen_runs.sort(key=lambda x: float(x.split('_')[-1].replace('b', '').replace('B', '')))
    
    if not qwen_runs:
        print(f"No Qwen runs found for scaling law heatmap, eval_dataset={eval_dataset}")
        return
    
    # Get all probe names from any Qwen run
    df_sample = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        run_name=qwen_runs[0] if qwen_runs else None,
        exclude_attention=True
    )
    
    probe_names = df_sample['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    probe_names.sort()
    
    # Create heatmap data
    heatmap_data = []
    model_sizes = []
    
    for run in qwen_runs:
        df = get_data_for_visualization(
            eval_dataset=eval_dataset,
            experiment='2-',
            run_name=run,
            exclude_attention=True
        )
        
        if df.empty:
            continue
        
        # Extract model size
        size_str = run.split('_')[-1].replace('b', '').replace('B', '')
        model_size = float(size_str)
        model_sizes.append(model_size)
        
        # Get data for n=1 positive sample
        n1_data = df[df['num_positive_samples'] == 1]
        
        row_data = []
        for probe in probe_names:
            probe_data = n1_data[n1_data['probe_name'] == probe]
            if not probe_data.empty:
                median_auc = probe_data['auc'].median()
                row_data.append(median_auc)
            else:
                row_data.append(np.nan)
        
        heatmap_data.append(row_data)
    
    if not heatmap_data:
        print(f"No data found for scaling law heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = np.array(heatmap_data)
    probe_labels = [get_probe_label(p) for p in probe_names]
    model_labels = [f'{size}B' for size in model_sizes]
    
    im = ax.imshow(heatmap_data.T, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels)
    ax.set_yticks(range(len(probe_labels)))
    ax.set_yticklabels([wrap_text(label, 20) for label in probe_labels])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median AUC')
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Probe Type')
    ax.set_title(f'Scaling Law Heatmap: AUC (n=1)\nEval Dataset: {wrap_text(eval_dataset)}')
    
    # Add text annotations
    for i in range(len(model_sizes)):
        for j in range(len(probe_names)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(i, j, f'{heatmap_data[i, j]:.3f}', 
                              ha='center', va='center', fontsize=8, color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_heatmap_recall(eval_dataset: str, save_path: Path):
    """Plot scaling law heatmap for Recall with probes vs model sizes."""
    setup_plot_style()
    
    # Get Qwen run names for scaling analysis
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    qwen_runs.sort(key=lambda x: float(x.split('_')[-1].replace('b', '').replace('B', '')))
    
    if not qwen_runs:
        print(f"No Qwen runs found for scaling law heatmap, eval_dataset={eval_dataset}")
        return
    
    # Get all probe names from any Qwen run
    df_sample = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        run_name=qwen_runs[0] if qwen_runs else None,
        exclude_attention=True
    )
    
    probe_names = df_sample['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    probe_names.sort()
    
    # Create heatmap data
    heatmap_data = []
    model_sizes = []
    
    for run in qwen_runs:
        df = get_data_for_visualization(
            eval_dataset=eval_dataset,
            experiment='2-',
            run_name=run,
            exclude_attention=True
        )
        
        if df.empty:
            continue
        
        # Extract model size
        size_str = run.split('_')[-1].replace('b', '').replace('B', '')
        model_size = float(size_str)
        model_sizes.append(model_size)
        
        # Get data for n=1 positive sample
        n1_data = df[df['num_positive_samples'] == 1]
        
        row_data = []
        for probe in probe_names:
            probe_data = n1_data[n1_data['probe_name'] == probe]
            if not probe_data.empty:
                median_recall = probe_data['recall'].median()
                row_data.append(median_recall)
            else:
                row_data.append(np.nan)
        
        heatmap_data.append(row_data)
    
    if not heatmap_data:
        print(f"No data found for scaling law heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = np.array(heatmap_data)
    probe_labels = [get_probe_label(p) for p in probe_names]
    model_labels = [f'{size}B' for size in model_sizes]
    
    im = ax.imshow(heatmap_data.T, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels)
    ax.set_yticks(range(len(probe_labels)))
    ax.set_yticklabels([wrap_text(label, 20) for label in probe_labels])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median Recall @ FPR=0.01')
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Probe Type')
    ax.set_title(f'Scaling Law Heatmap: Recall (n=1)\nEval Dataset: {wrap_text(eval_dataset)}')
    
    # Add text annotations
    for i in range(len(model_sizes)):
        for j in range(len(probe_names)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(i, j, f'{heatmap_data[i, j]:.3f}', 
                              ha='center', va='center', fontsize=8, color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_heatmap_auc(eval_dataset: str, save_path: Path):
    """Plot LLM upsampling heatmap for AUC with probes vs upsampling ratios."""
    setup_plot_style()
    
    # Get data for experiment 3 (LLM upsampling)
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling heatmap, eval_dataset={eval_dataset}")
        return
    
    # Get unique probe names and upsampling ratios
    probe_names = df['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    probe_names.sort()
    
    upsampling_ratios = sorted(df['llm_upsampling_ratio'].dropna().unique())
    
    if not upsampling_ratios:
        print(f"No upsampling ratios found for heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap data
    heatmap_data = []
    
    for ratio in upsampling_ratios:
        ratio_data = df[df['llm_upsampling_ratio'] == ratio]
        
        if ratio_data.empty:
            continue
        
        # Get data for n=1 positive sample
        n1_data = ratio_data[ratio_data['num_positive_samples'] == 1]
        
        row_data = []
        for probe in probe_names:
            probe_data = n1_data[n1_data['probe_name'] == probe]
            if not probe_data.empty:
                median_auc = probe_data['auc'].median()
                row_data.append(median_auc)
            else:
                row_data.append(np.nan)
        
        heatmap_data.append(row_data)
    
    if not heatmap_data:
        print(f"No data found for LLM upsampling heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = np.array(heatmap_data)
    probe_labels = [get_probe_label(p) for p in probe_names]
    ratio_labels = [f'{ratio}x' for ratio in upsampling_ratios]
    
    im = ax.imshow(heatmap_data.T, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(ratio_labels)))
    ax.set_xticklabels(ratio_labels)
    ax.set_yticks(range(len(probe_labels)))
    ax.set_yticklabels([wrap_text(label, 20) for label in probe_labels])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median AUC')
    
    ax.set_xlabel('Upsampling Ratio')
    ax.set_ylabel('Probe Type')
    ax.set_title(f'LLM Upsampling Heatmap: AUC (n=1)\nEval Dataset: {wrap_text(eval_dataset)}')
    
    # Add text annotations
    for i in range(len(upsampling_ratios)):
        for j in range(len(probe_names)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(i, j, f'{heatmap_data[i, j]:.3f}', 
                              ha='center', va='center', fontsize=8, color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_heatmap_recall(eval_dataset: str, save_path: Path):
    """Plot LLM upsampling heatmap for Recall with probes vs upsampling ratios."""
    setup_plot_style()
    
    # Get data for experiment 3 (LLM upsampling)
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling heatmap, eval_dataset={eval_dataset}")
        return
    
    # Get unique probe names and upsampling ratios
    probe_names = df['probe_name'].unique()
    probe_names = [p for p in probe_names if p and 'attention' not in p]
    probe_names.sort()
    
    upsampling_ratios = sorted(df['llm_upsampling_ratio'].dropna().unique())
    
    if not upsampling_ratios:
        print(f"No upsampling ratios found for heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap data
    heatmap_data = []
    
    for ratio in upsampling_ratios:
        ratio_data = df[df['llm_upsampling_ratio'] == ratio]
        
        if ratio_data.empty:
            continue
        
        # Get data for n=1 positive sample
        n1_data = ratio_data[ratio_data['num_positive_samples'] == 1]
        
        row_data = []
        for probe in probe_names:
            probe_data = n1_data[n1_data['probe_name'] == probe]
            if not probe_data.empty:
                median_recall = probe_data['recall'].median()
                row_data.append(median_recall)
            else:
                row_data.append(np.nan)
        
        heatmap_data.append(row_data)
    
    if not heatmap_data:
        print(f"No data found for LLM upsampling heatmap, eval_dataset={eval_dataset}")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = np.array(heatmap_data)
    probe_labels = [get_probe_label(p) for p in probe_names]
    ratio_labels = [f'{ratio}x' for ratio in upsampling_ratios]
    
    im = ax.imshow(heatmap_data.T, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(ratio_labels)))
    ax.set_xticklabels(ratio_labels)
    ax.set_yticks(range(len(probe_labels)))
    ax.set_yticklabels([wrap_text(label, 20) for label in probe_labels])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Median Recall @ FPR=0.01')
    
    ax.set_xlabel('Upsampling Ratio')
    ax.set_ylabel('Probe Type')
    ax.set_title(f'LLM Upsampling Heatmap: Recall (n=1)\nEval Dataset: {wrap_text(eval_dataset)}')
    
    # Add text annotations
    for i in range(len(upsampling_ratios)):
        for j in range(len(probe_names)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(i, j, f'{heatmap_data[i, j]:.3f}', 
                              ha='center', va='center', fontsize=8, color='white')
    
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
        # Experiment 2 best probes - AUC
        plot_experiment_2_best_probes_auc(
            eval_dataset, 
            main_dir / f"experiment_2_best_probes_auc_{eval_dataset}.png"
        )
        
        # Experiment 2 best probes - Recall
        plot_experiment_2_best_probes_recall(
            eval_dataset, 
            main_dir / f"experiment_2_best_probes_recall_{eval_dataset}.png"
        )
        
        # Experiment 4 best probes - AUC
        plot_experiment_4_best_probes_auc(
            eval_dataset, 
            main_dir / f"experiment_4_best_probes_auc_{eval_dataset}.png"
        )
        
        # Experiment 4 best probes - Recall
        plot_experiment_4_best_probes_recall(
            eval_dataset, 
            main_dir / f"experiment_4_best_probes_recall_{eval_dataset}.png"
        )
        
        # LLM upsampling aggregated - AUC
        plot_llm_upsampling_aggregated_auc(
            eval_dataset, 
            main_dir / f"llm_upsampling_aggregated_auc_{eval_dataset}.png"
        )
        
        # LLM upsampling aggregated - Recall
        plot_llm_upsampling_aggregated_recall(
            eval_dataset, 
            main_dir / f"llm_upsampling_aggregated_recall_{eval_dataset}.png"
        )
        
        # Scaling law aggregated - AUC
        plot_scaling_law_aggregated_auc(
            eval_dataset, 
            main_dir / f"scaling_law_aggregated_auc_{eval_dataset}.png"
        )
        
        # Scaling law aggregated - Recall
        plot_scaling_law_aggregated_recall(
            eval_dataset, 
            main_dir / f"scaling_law_aggregated_recall_{eval_dataset}.png"
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
    
    print("Generating heatmap visualizations...")
    
    for eval_dataset in eval_datasets:
        # Scaling Law Heatmap - AUC
        plot_scaling_law_heatmap_auc(
            eval_dataset,
            other_dir / f"scaling_law_heatmap_auc_{eval_dataset}.png"
        )
        
        # Scaling Law Heatmap - Recall
        plot_scaling_law_heatmap_recall(
            eval_dataset,
            other_dir / f"scaling_law_heatmap_recall_{eval_dataset}.png"
        )
        
        # LLM Upsampling Heatmap - AUC
        plot_llm_upsampling_heatmap_auc(
            eval_dataset,
            other_dir / f"llm_upsampling_heatmap_auc_{eval_dataset}.png"
        )
        
        # LLM Upsampling Heatmap - Recall
        plot_llm_upsampling_heatmap_recall(
            eval_dataset,
            other_dir / f"llm_upsampling_heatmap_recall_{eval_dataset}.png"
        )
    
    print("All visualizations generated successfully!")
    print(f"Main plots saved to: {main_dir}")
    print(f"Other plots saved to: {other_dir}")


if __name__ == "__main__":
    generate_all_visualizations()
