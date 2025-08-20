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


def plot_probe_subplots_unified(experiment: str, eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Create subplot visualization for all probes - unified function for AUC and Recall."""
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
    
    # Define the probe names we want to include (same as other subplots)
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
    ]
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
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
    
    for i, probe in enumerate(probe_names[:12]):  # Limit to 12 probes
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
            
            ax.plot(x_values, y_means, 'o-', color=color, markersize=4, linewidth=1.5)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Training Size')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(y_min_default, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    
    # Hide unused subplots
    for i in range(len(probe_names), 12):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Experiment {experiment}: Individual Probe Performance {title_suffix}\nEval Dataset: {wrap_text(eval_dataset)}', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_subplots_unified(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Create subplot visualization for LLM upsampling across all probes - unified function."""
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
    
    # Define the probe names we want to include (same as heatmaps)
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'linear_last', 'linear_max', 'linear_mean', 'linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
    ]
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC'
        y_min_default = 0.6
        title_suffix = '(AUC)'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01'
        y_min_default = 0.0
        title_suffix = '(Recall @ FPR=0.01)'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    for i, probe in enumerate(probe_names[:12]):  # Limit to 12 probes
        ax = axes[i]
        
        # Get data for this probe
        probe_data = df[df['probe_name'] == probe]
        
        if not probe_data.empty:
            # Get unique upsampling ratios
            upsampling_ratios = sorted(probe_data['llm_upsampling_ratio'].dropna().unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(upsampling_ratios)))
            
            for j, ratio in enumerate(upsampling_ratios):
                ratio_data = probe_data[probe_data['llm_upsampling_ratio'] == ratio]
                if not ratio_data.empty:
                    # Group by positive samples and calculate median
                    grouped = ratio_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
                    grouped = grouped.sort_values('num_positive_samples')
                    
                    x_values = grouped['num_positive_samples'].values
                    y_medians = [np.median(values) for values in grouped[metric]]
                    
                    ax.plot(x_values, y_medians, 'o-', color=colors[j], 
                           label=f'{ratio}x', markersize=3, linewidth=1)
            
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Positive Samples')
            ax.set_ylabel(ylabel)
            
            # Set dynamic y-axis limits
            all_y_values = []
            for ratio in upsampling_ratios:
                ratio_data = probe_data[probe_data['llm_upsampling_ratio'] == ratio]
                if not ratio_data.empty:
                    grouped = ratio_data.groupby('num_positive_samples')[metric].apply(list).reset_index()
                    medians = [np.median(values) for values in grouped[metric]]
                    all_y_values.extend(medians)
            
            if all_y_values:
                y_min = max(y_min_default, min(all_y_values) - 0.05)
                y_max = min(1.0, max(all_y_values) + 0.05)
                ax.set_ylim(y_min, y_max)
            
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
    
    plt.suptitle(f'LLM Upsampling: Individual Probe Performance {title_suffix}\nEval Dataset: {wrap_text(eval_dataset)}', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_subplots_unified(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Create subplot visualization for scaling law across all probes - unified function."""
    setup_plot_style()
    
    # Get Qwen run names
    qwen_runs = [run for run in get_run_names() if 'qwen' in run.lower()]
    qwen_runs.sort(key=lambda x: float(x.split('_')[-1].replace('b', '').replace('B', '')))
    
    if not qwen_runs:
        print(f"No Qwen runs found for scaling law subplots, eval_dataset={eval_dataset}")
        return
    
    # Define the probe names we want to include (same as heatmaps)
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'linear_last', 'linear_max', 'linear_mean', 'linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
    ]
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
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
    
    for i, probe in enumerate(probe_names[:12]):  # Limit to 12 probes
        ax = axes[i]
        
        # Collect data for this probe across all Qwen models
        model_sizes = []
        metric_values = []
        
        for run in qwen_runs:
            df = get_data_for_visualization(
                eval_dataset=eval_dataset,
                experiment='2-',
                run_name=run,
                probe_name=probe
            )
            
            if not df.empty:
                # Use individual values instead of median
                metric_values.extend(df[metric].tolist())
                
                # Extract model size
                size_str = run.split('_')[-1].replace('b', '').replace('B', '')
                model_size = float(size_str)
                model_sizes.extend([model_size] * len(df))
        
        if model_sizes and metric_values:
            # Plot individual points
            ax.scatter(model_sizes, metric_values, alpha=0.6, s=20)
            
            # Also plot median line
            unique_sizes = sorted(set(model_sizes))
            medians = []
            for size in unique_sizes:
                size_indices = [j for j, s in enumerate(model_sizes) if s == size]
                size_values = [metric_values[j] for j in size_indices]
                medians.append(np.median(size_values))
            
            ax.plot(unique_sizes, medians, 'o-', color='red', linewidth=2, markersize=6, label='Median')
            
            ax.set_title(wrap_text(get_probe_label(probe), 25))
            ax.set_xlabel('Model Size (B)')
            ax.set_ylabel(ylabel)
            
            # Set dynamic y-axis limits
            if metric_values:
                y_min = max(y_min_default, min(metric_values) - 0.05)
                y_max = min(1.0, max(metric_values) + 0.05)
                ax.set_ylim(y_min, y_max)
            
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            if i == 0:  # Only show legend on first subplot
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_probe_label(probe), 25))
    
    # Hide unused subplots
    for i in range(len(probe_names), 12):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Scaling Law: Individual Probe Performance {title_suffix}\nEval Dataset: {wrap_text(eval_dataset)}', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Convenience functions that call the unified versions
def plot_probe_subplots_auc(experiment: str, eval_dataset: str, save_path: Path):
    """Create subplot visualization for all probes - AUC."""
    plot_probe_subplots_unified(experiment, eval_dataset, save_path, 'auc')


def plot_probe_subplots_recall(experiment: str, eval_dataset: str, save_path: Path):
    """Create subplot visualization for all probes - Recall."""
    plot_probe_subplots_unified(experiment, eval_dataset, save_path, 'recall')


def plot_llm_upsampling_subplots_auc(eval_dataset: str, save_path: Path):
    """Create subplot visualization for LLM upsampling across all probes - AUC."""
    plot_llm_upsampling_subplots_unified(eval_dataset, save_path, 'auc')


def plot_llm_upsampling_subplots_recall(eval_dataset: str, save_path: Path):
    """Create subplot visualization for LLM upsampling across all probes - Recall."""
    plot_llm_upsampling_subplots_unified(eval_dataset, save_path, 'recall')


def plot_scaling_law_subplots_auc(eval_dataset: str, save_path: Path):
    """Create subplot visualization for scaling law across all probes - AUC."""
    plot_scaling_law_subplots_unified(eval_dataset, save_path, 'auc')


def plot_scaling_law_subplots_recall(eval_dataset: str, save_path: Path):
    """Create subplot visualization for scaling law across all probes - Recall."""
    plot_scaling_law_subplots_unified(eval_dataset, save_path, 'recall')


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
        plot_probe_subplots_auc(
            '2-', eval_dataset, 
            other_dir / f"experiment_2_subplots_auc_{eval_dataset}.png"
        )
        
        # Experiment 2 subplots
        plot_probe_subplots_recall(
            '2-', eval_dataset, 
            other_dir / f"experiment_2_subplots_recall_{eval_dataset}.png"
        )
        
        # Experiment 4 subplots
        plot_probe_subplots_auc(
            '4-', eval_dataset, 
            other_dir / f"experiment_4_subplots_auc_{eval_dataset}.png"
        )
        
        # Experiment 4 subplots
        plot_probe_subplots_recall(
            '4-', eval_dataset, 
            other_dir / f"experiment_4_subplots_recall_{eval_dataset}.png"
        )
        
        # LLM upsampling subplots
        plot_llm_upsampling_subplots_auc(
            eval_dataset, 
            other_dir / f"llm_upsampling_subplots_auc_{eval_dataset}.png"
        )
        
        # LLM upsampling subplots
        plot_llm_upsampling_subplots_recall(
            eval_dataset, 
            other_dir / f"llm_upsampling_subplots_recall_{eval_dataset}.png"
        )
        
        # Scaling law subplots - AUC
        plot_scaling_law_subplots_auc(
            eval_dataset, 
            other_dir / f"scaling_law_subplots_auc_{eval_dataset}.png"
        )
        
        # Scaling law subplots - Recall
        plot_scaling_law_subplots_recall(
            eval_dataset, 
            other_dir / f"scaling_law_subplots_recall_{eval_dataset}.png"
        )
    
    print("Generating heatmap visualizations...")
    
    for eval_dataset in eval_datasets:
        # Scaling Law Heatmap - AUC
        plot_scaling_law_heatmap_auc(
            eval_dataset,
            main_dir / f"scaling_law_heatmap_auc_{eval_dataset}.png"
        )
        
        # Scaling Law Heatmap - Recall
        plot_scaling_law_heatmap_recall(
            eval_dataset,
            main_dir / f"scaling_law_heatmap_recall_{eval_dataset}.png"
        )
        
        # LLM Upsampling Heatmap - AUC
        plot_llm_upsampling_heatmap_auc(
            eval_dataset,
            main_dir / f"llm_upsampling_heatmap_auc_{eval_dataset}.png"
        )
        
        # LLM Upsampling Heatmap - Recall
        plot_llm_upsampling_heatmap_recall(
            eval_dataset,
            main_dir / f"llm_upsampling_heatmap_recall_{eval_dataset}.png"
        )
    
    print("All visualizations generated successfully!")
    print(f"Main plots saved to: {main_dir}")
    print(f"Other plots saved to: {other_dir}")


def get_probe_names_for_model(run_name: str) -> list:
    """
    Get the appropriate probe names based on the model type.
    For Gemma models, we have 8 SAE probes (2 different SAEs for each aggregation).
    For Qwen models, we have 4 SAE probes.
    """
    if 'gemma' in run_name.lower():
        # Gemma has 8 SAE probes (2 different SAEs for each aggregation)
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'linear_last', 'linear_max', 'linear_mean', 'linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax',
            'sae2_last', 'sae2_max', 'sae2_mean', 'sae2_softmax'  # Second SAE variant
        ]
    else:
        # Qwen and other models have 4 SAE probes
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'linear_last', 'linear_max', 'linear_mean', 'linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    return probe_names


def get_probe_label_with_config(probe_name: str, run_name: str = None) -> str:
    """
    Get a more descriptive probe label that includes configuration information.
    This function can be extended to include SAE widths, L0s, etc. from a config file.
    """
    base_label = get_probe_label(probe_name)
    
    # For now, we'll use a simple approach. In the future, this could read from a config file
    if 'sae' in probe_name and run_name:
        if 'gemma' in run_name.lower():
            if 'sae2' in probe_name:
                # Second SAE variant for Gemma
                return f"{base_label} (SAE2)"
            else:
                # First SAE variant for Gemma
                return f"{base_label} (SAE1)"
        else:
            # Qwen SAE
            return f"{base_label} (Qwen)"
    
    return base_label


def create_probe_config_file():
    """
    Create a configuration file for probe labels and configurations.
    This can be manually edited to include SAE widths, L0s, etc.
    """
    config = {
        "probe_configs": {
            "gemma_9b": {
                "sae_last": {"width": 16384, "l0": 0.408, "label": "SAE (last, 16k, L0=0.408)"},
                "sae_max": {"width": 16384, "l0": 0.408, "label": "SAE (max, 16k, L0=0.408)"},
                "sae_mean": {"width": 16384, "l0": 0.408, "label": "SAE (mean, 16k, L0=0.408)"},
                "sae_softmax": {"width": 16384, "l0": 0.408, "label": "SAE (softmax, 16k, L0=0.408)"},
                "sae2_last": {"width": 8192, "l0": 0.204, "label": "SAE (last, 8k, L0=0.204)"},
                "sae2_max": {"width": 8192, "l0": 0.204, "label": "SAE (max, 8k, L0=0.204)"},
                "sae2_mean": {"width": 8192, "l0": 0.204, "label": "SAE (mean, 8k, L0=0.204)"},
                "sae2_softmax": {"width": 8192, "l0": 0.204, "label": "SAE (softmax, 8k, L0=0.204)"}
            },
            "qwen_0.5b": {
                "sae_last": {"width": 4096, "l0": 0.1, "label": "SAE (last, 4k, L0=0.1)"},
                "sae_max": {"width": 4096, "l0": 0.1, "label": "SAE (max, 4k, L0=0.1)"},
                "sae_mean": {"width": 4096, "l0": 0.1, "label": "SAE (mean, 4k, L0=0.1)"},
                "sae_softmax": {"width": 4096, "l0": 0.1, "label": "SAE (softmax, 4k, L0=0.1)"}
            },
            "qwen_1.5b": {
                "sae_last": {"width": 4096, "l0": 0.1, "label": "SAE (last, 4k, L0=0.1)"},
                "sae_max": {"width": 4096, "l0": 0.1, "label": "SAE (max, 4k, L0=0.1)"},
                "sae_mean": {"width": 4096, "l0": 0.1, "label": "SAE (mean, 4k, L0=0.1)"},
                "sae_softmax": {"width": 4096, "l0": 0.1, "label": "SAE (softmax, 4k, L0=0.1)"}
            },
            "qwen_4b": {
                "sae_last": {"width": 4096, "l0": 0.1, "label": "SAE (last, 4k, L0=0.1)"},
                "sae_max": {"width": 4096, "l0": 0.1, "label": "SAE (max, 4k, L0=0.1)"},
                "sae_mean": {"width": 4096, "l0": 0.1, "label": "SAE (mean, 4k, L0=0.1)"},
                "sae_softmax": {"width": 4096, "l0": 0.1, "label": "SAE (softmax, 4k, L0=0.1)"}
            },
            "qwen_7b": {
                "sae_last": {"width": 4096, "l0": 0.1, "label": "SAE (last, 4k, L0=0.1)"},
                "sae_max": {"width": 4096, "l0": 0.1, "label": "SAE (max, 4k, L0=0.1)"},
                "sae_mean": {"width": 4096, "l0": 0.1, "label": "SAE (mean, 4k, L0=0.1)"},
                "sae_softmax": {"width": 4096, "l0": 0.1, "label": "SAE (softmax, 4k, L0=0.1)"}
            },
            "qwen_14b": {
                "sae_last": {"width": 4096, "l0": 0.1, "label": "SAE (last, 4k, L0=0.1)"},
                "sae_max": {"width": 4096, "l0": 0.1, "label": "SAE (max, 4k, L0=0.1)"},
                "sae_mean": {"width": 4096, "l0": 0.1, "label": "SAE (mean, 4k, L0=0.1)"},
                "sae_softmax": {"width": 4096, "l0": 0.1, "label": "SAE (softmax, 4k, L0=0.1)"}
            },
            "qwen_72b": {
                "sae_last": {"width": 4096, "l0": 0.1, "label": "SAE (last, 4k, L0=0.1)"},
                "sae_max": {"width": 4096, "l0": 0.1, "label": "SAE (max, 4k, L0=0.1)"},
                "sae_mean": {"width": 4096, "l0": 0.1, "label": "SAE (mean, 4k, L0=0.1)"},
                "sae_softmax": {"width": 4096, "l0": 0.1, "label": "SAE (softmax, 4k, L0=0.1)"}
            }
        }
    }
    
    import json
    config_path = Path("src/visualize/probe_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created probe configuration file: {config_path}")
    print("You can edit this file to update SAE widths, L0s, and labels for different models.")


def load_probe_config():
    """Load probe configuration from JSON file."""
    config_path = Path("src/visualize/probe_config.json")
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def get_detailed_probe_label(probe_name: str, run_name: str = None) -> str:
    """
    Get a detailed probe label using the configuration file.
    Falls back to the basic label if config is not available.
    """
    config = load_probe_config()
    if not config or not run_name:
        return get_probe_label(probe_name)
    
    # Find the appropriate model config
    model_key = None
    for key in config["probe_configs"].keys():
        if key in run_name.lower():
            model_key = key
            break
    
    if model_key and probe_name in config["probe_configs"][model_key]:
        return config["probe_configs"][model_key][probe_name]["label"]
    
    return get_probe_label(probe_name)


if __name__ == "__main__":
    generate_all_visualizations()
