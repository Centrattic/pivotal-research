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


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
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


def load_probe_config():
    """Load probe configuration from JSON file."""
    try:
        import json
        config_path = Path("src/visualize/probe_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Probe config file not found at {config_path}")
            return {}
    except Exception as e:
        print(f"Warning: Could not load probe config: {e}")
        return {}


def get_detailed_probe_label(probe_name: str, run_name: str = None) -> str:
    """
    Get detailed probe label with SAE width and L0 information.
    Falls back to basic label if config not available.
    """
    # Load probe configuration
    config = load_probe_config()
    
    # If no config or not an SAE probe, use basic label
    if not config or 'sae' not in probe_name:
        return get_probe_label(probe_name)
    
    # Try to find the model in the config
    model_key = None
    if run_name:
        if 'gemma' in run_name.lower():
            model_key = 'gemma_9b'
        elif 'qwen' in run_name.lower():
            # Extract model size from run name
            size_match = re.search(r'qwen_([0-9.]+)b', run_name, re.IGNORECASE)
            if size_match:
                size = size_match.group(1)
                model_key = f'qwen_{size}b'
    
    # If we found a model key and it has this probe, use the detailed label
    if model_key and model_key in config.get('probe_configs', {}):
        probe_configs = config['probe_configs'][model_key]
        if probe_name in probe_configs:
            return probe_configs[probe_name]['label']
    
    # Fallback to basic label
    return get_probe_label(probe_name)


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
    
    # Debug: Print data before filtering
    print(f"\n=== EXPERIMENT 2 AUC PLOT DEBUG for {eval_dataset} ===")
    print(f"Before filtering - Total rows: {len(df)}")
    print(f"Before filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    
    # Debug: Check sample counts
    print(f"Before filtering - num_positive_samples: {df['num_positive_samples'].value_counts().head(10).to_dict()}")
    print(f"Before filtering - num_negative_samples: {df['num_negative_samples'].value_counts().head(10).to_dict()}")
    
    # Apply main plot filters (Gemma SAE topk=1024, linear C=1.0)
    df = apply_main_plot_filters(df)
    
    if df.empty:
        print(f"No data found after applying filters for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Debug: Print data after filtering
    print(f"After filtering - Total rows: {len(df)}")
    print(f"After filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    print(f"After filtering - num_positive_samples: {df['num_positive_samples'].value_counts().to_dict()}")
    print(f"After filtering - num_negative_samples: {df['num_negative_samples'].value_counts().to_dict()}")
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Debug: Print best probes and their data counts
    print(f"Best probes selected: {best_probes}")
    for probe in best_probes:
        probe_data = df[df['probe_name'] == probe]
        print(f"  {probe}: {len(probe_data)} data rows")
        if not probe_data.empty:
            print(f"    Sample counts: {probe_data['num_positive_samples'].value_counts().to_dict()}")
            print(f"    Seeds: {probe_data['seed'].value_counts().to_dict()}")
    
    print("=== END DEBUG ===\n")
    
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
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    # Get number of negative examples for title
    num_negative = df['num_negative_samples'].iloc[0] if not df.empty and 'num_negative_samples' in df.columns else None
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('AUC')
    if num_negative is not None:
        ax.set_title(f'Unbalanced ({num_negative} negative examples)')
    else:
        ax.set_title(f'Unbalanced')
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
    
    # Debug: Print data before filtering
    print(f"\n=== EXPERIMENT 2 RECALL PLOT DEBUG for {eval_dataset} ===")
    print(f"Before filtering - Total rows: {len(df)}")
    print(f"Before filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    print(f"Before filtering - num_positive_samples: {df['num_positive_samples'].value_counts().head(10).to_dict()}")
    print(f"Before filtering - num_negative_samples: {df['num_negative_samples'].value_counts().head(10).to_dict()}")
    
    # Apply main plot filters (Gemma SAE topk=1024, linear C=1.0)
    df = apply_main_plot_filters(df)
    
    if df.empty:
        print(f"No data found after applying filters for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Debug: Print data after filtering
    print(f"After filtering - Total rows: {len(df)}")
    print(f"After filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    print(f"After filtering - num_positive_samples: {df['num_positive_samples'].value_counts().to_dict()}")
    print(f"After filtering - num_negative_samples: {df['num_negative_samples'].value_counts().to_dict()}")
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 2, eval_dataset={eval_dataset}")
        return
    
    # Debug: Print best probes and their data counts
    print(f"Best probes selected: {best_probes}")
    for probe in best_probes:
        probe_data = df[df['probe_name'] == probe]
        print(f"  {probe}: {len(probe_data)} data rows")
        if not probe_data.empty:
            print(f"    Sample counts: {probe_data['num_positive_samples'].value_counts().to_dict()}")
            print(f"    Seeds: {probe_data['seed'].value_counts().to_dict()}")
    
    print("=== END DEBUG ===\n")
    
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
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    # Get number of negative examples for title
    num_negative = df['num_negative_samples'].iloc[0] if not df.empty and 'num_negative_samples' in df.columns else None
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Recall @ FPR=0.01')
    if num_negative is not None:
        ax.set_title(f'Unbalanced ({num_negative} negative examples)')
    else:
        ax.set_title(f'Unbalanced')
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
    
    # Debug: show SAE presence before filtering
    print(f"\n=== EXPERIMENT 4 AUC PLOT DEBUG for {eval_dataset} ===")
    print(f"Before filtering - Total rows: {len(df)}")
    try:
        print(f"Before filtering - SAE rows: {len(df[df['probe_name'].str.contains('sae', na=False)])}")
        print(f"Before filtering - Example SAE filenames:")
        print(df[df['probe_name'].str.contains('sae', na=False)]['filename'].head(10).to_list())
    except Exception:
        pass
    
    # Apply main plot filters (Gemma SAE topk=1024, linear C=1.0)
    df = apply_main_plot_filters(df)
    
    if df.empty:
        print(f"No data found after applying filters for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Debug: show SAE presence after filtering
    try:
        print(f"After filtering - Total rows: {len(df)}")
        print(f"After filtering - SAE rows: {len(df[df['probe_name'].str.contains('sae', na=False)])}")
        print(f"After filtering - Example SAE filenames:")
        print(df[df['probe_name'].str.contains('sae', na=False)]['filename'].head(10).to_list())
    except Exception:
        pass
    
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
        
        # Debug: list contributing filenames per probe
        try:
            contributing_files = sorted(probe_data['filename'].unique().tolist())
            print(f"[files] {probe}: {len(contributing_files)} files")
            for f in contributing_files:
                print(f"  - {f}")
        except Exception:
            pass
        
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
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of training examples per class\n(x negative, x positive)')
    ax.set_ylabel('AUC')
    ax.set_title(f'Balanced (equal positive and negative samples)')
    
    # Only set log scale if we have positive x values
    if all_y_values and len(all_y_values) > 0:
        # Check if we have any positive x values from the data
        all_x_values = []
        for probe in best_probes:
            probe_data = df[df['probe_name'] == probe]
            if not probe_data.empty:
                all_x_values.extend(probe_data['num_positive_samples'].dropna().values)
        
        if all_x_values and min(all_x_values) > 0:
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
    
    # Debug: show SAE presence before filtering
    print(f"\n=== EXPERIMENT 4 RECALL PLOT DEBUG for {eval_dataset} ===")
    print(f"Before filtering - Total rows: {len(df)}")
    try:
        print(f"Before filtering - SAE rows: {len(df[df['probe_name'].str.contains('sae', na=False)])}")
        print(f"Before filtering - Example SAE filenames:")
        print(df[df['probe_name'].str.contains('sae', na=False)]['filename'].head(10).to_list())
    except Exception:
        pass
    
    # Debug: Print data before filtering
    print(f"\n=== EXPERIMENT 4 RECALL PLOT DEBUG for {eval_dataset} ===")
    print(f"Before filtering - Total rows: {len(df)}")
    print(f"Before filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    print(f"Before filtering - num_positive_samples: {df['num_positive_samples'].value_counts().head(10).to_dict()}")
    print(f"Before filtering - num_negative_samples: {df['num_negative_samples'].value_counts().head(10).to_dict()}")
    
    # Apply main plot filters (Gemma SAE topk=1024, linear C=1.0)
    df = apply_main_plot_filters(df)
    
    if df.empty:
        print(f"No data found after applying filters for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Debug: show SAE presence after filtering
    try:
        print(f"After filtering - Total rows: {len(df)}")
        print(f"After filtering - SAE rows: {len(df[df['probe_name'].str.contains('sae', na=False)])}")
        print(f"After filtering - Example SAE filenames:")
        print(df[df['probe_name'].str.contains('sae', na=False)]['filename'].head(10).to_list())
    except Exception:
        pass
    
    # Debug: Print data after filtering
    print(f"After filtering - Total rows: {len(df)}")
    print(f"After filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    print(f"After filtering - num_positive_samples: {df['num_positive_samples'].value_counts().to_dict()}")
    print(f"After filtering - num_negative_samples: {df['num_negative_samples'].value_counts().to_dict()}")
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 4, eval_dataset={eval_dataset}")
        return
    
    # Debug: Print best probes and their data counts
    print(f"Best probes selected: {best_probes}")
    for probe in best_probes:
        probe_data = df[df['probe_name'] == probe]
        print(f"  {probe}: {len(probe_data)} data rows")
        if not probe_data.empty:
            print(f"    Sample counts: {probe_data['num_positive_samples'].value_counts().to_dict()}")
            print(f"    Seeds: {probe_data['seed'].value_counts().to_dict()}")
    
    print("=== END DEBUG ===\n")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    all_y_values = []  # Collect all y values for proper ylim
    
    for i, probe in enumerate(best_probes):
        probe_data = df[df['probe_name'] == probe]
        
        if probe_data.empty:
            continue
        
        # Debug: list contributing filenames per probe
        try:
            contributing_files = sorted(probe_data['filename'].unique().tolist())
            print(f"[files] {probe}: {len(contributing_files)} files")
            for f in contributing_files:
                print(f"  - {f}")
        except Exception:
            pass
        
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
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Number of training examples per class\n(x negative, x positive)')
    ax.set_ylabel('Recall @ FPR=0.01')
    ax.set_title(f'Balanced (equal positive and negative samples)')
    
    # Only set log scale if we have positive x values
    if all_y_values and len(all_y_values) > 0:
        # Check if we have any positive x values from the data
        all_x_values = []
        for probe in best_probes:
            probe_data = df[df['probe_name'] == probe]
            if not probe_data.empty:
                all_x_values.extend(probe_data['num_positive_samples'].dropna().values)
        
        if all_x_values and min(all_x_values) > 0:
            ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_aggregated(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Plot LLM upsampling with median performance across all probes."""
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
    fig, ax = plt.subplots(figsize=(6, 4))
    
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
                label=f'{ratio_int}x', linewidth=2, markersize=4)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for LLM upsampling {metric}, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
        title_suffix = 'AUC'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01 (across all probes)'
        title_suffix = 'Recall'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    # Format eval_dataset for cleaner title
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    ax.set_title(f'LLM Upsampling: Median {title_suffix} Performance\n{formatted_dataset} Dataset')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend(title='Upsampling \nFactor')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_aggregated_auc(eval_dataset: str, save_path: Path):
    """Plot LLM upsampling with median AUC performance across all probes."""
    plot_llm_upsampling_aggregated(eval_dataset, save_path, metric='auc')


def plot_llm_upsampling_aggregated_recall(eval_dataset: str, save_path: Path):
    """Plot LLM upsampling with median Recall performance across all probes."""
    plot_llm_upsampling_aggregated(eval_dataset, save_path, metric='recall')


def plot_scaling_law_aggregated(eval_dataset: str, save_path: Path, metric: str = 'auc'):
    """Plot scaling law with median performance across all probes."""
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
        grouped = df.groupby('num_positive_samples')[metric].apply(list).reset_index()
        grouped = grouped.sort_values('num_positive_samples')
        
        if grouped.empty:
            continue
        
        x_values = grouped['num_positive_samples'].values
        y_medians = [np.median(metric_list) for metric_list in grouped[metric]]
        all_y_values.extend(y_medians)
        
        # Plot line
        ax.plot(x_values, y_medians, 'o-', color=colors[i], 
                label=f'{model_size}B', linewidth=2, markersize=4)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for scaling law {metric}, eval_dataset={eval_dataset}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
        title_suffix = 'AUC'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01 (across all probes)'
        title_suffix = 'Recall'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Scaling across Qwen model sizes: {title_suffix}\nEval Dataset: {wrap_text(eval_dataset)}')
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
    plot_scaling_law_aggregated(eval_dataset, save_path, metric='auc')


def plot_scaling_law_aggregated_recall(eval_dataset: str, save_path: Path):
    """Plot scaling law with median Recall performance across all probes."""
    plot_scaling_law_aggregated(eval_dataset, save_path, metric='recall')


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
    # For subplots, we need to determine if this is a Gemma experiment
    # Check if any run names contain 'gemma'
    run_names = get_run_names()
    gemma_runs = [run for run in run_names if 'gemma' in run.lower()]
    
    # Check if this specific experiment has Gemma data
    experiment_runs = [run for run in run_names if experiment in run]
    experiment_has_gemma = any('gemma' in run.lower() for run in experiment_runs)
    
    if experiment_has_gemma:
        # Gemma has 12 probes total (4 act_sim + 4 linear + 4 SAE) - no more sae2_
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    else:
        # Qwen and other models have 12 probes total (4 act_sim + 4 linear + 4 SAE)
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    
    # Create subplot grid based on number of probes
    # All models now have 12 probes (4 act_sim + 4 linear + 4 SAE)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
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
    
    for i, probe in enumerate(probe_names):  # Use all probes
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
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
            ax.set_xlabel('Training Size')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(y_min_default, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    
    # Hide unused subplots
    max_subplots = 12  # All models now have 12 probes
    for i in range(len(probe_names), max_subplots):
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
    # For subplots, we need to determine if this is a Gemma experiment
    # Check if any run names contain 'gemma'
    run_names = get_run_names()
    gemma_runs = [run for run in run_names if 'gemma' in run.lower()]
    
    # Check if this specific experiment (LLM upsampling) has Gemma data
    experiment_runs = [run for run in run_names if '3-' in run]  # Experiment 3 is LLM upsampling
    experiment_has_gemma = any('gemma' in run.lower() for run in experiment_runs)
    
    if experiment_has_gemma:
        # Gemma has 12 probes total (4 act_sim + 4 linear + 4 SAE) - no more sae2_
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    else:
        # Qwen and other models have 12 probes total (4 act_sim + 4 linear + 4 SAE)
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    
    # Create subplot grid based on number of probes
    # All models now have 12 probes (4 act_sim + 4 linear + 4 SAE)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
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
    
    for i, probe in enumerate(probe_names):  # Use all probes
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
            
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
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
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
    
    # Hide unused subplots
    max_subplots = 12  # All models now have 12 probes
    for i in range(len(probe_names), max_subplots):
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
    # For scaling law, we only have Qwen models, so we use 12 probes
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
    ]
    
    # Create 3x4 subplot grid for Qwen models
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
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
    
    for i, probe in enumerate(probe_names):  # Use all probes
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
            
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
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
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
    
    # Hide unused subplots
    max_subplots = 12  # Scaling law only has Qwen models
    for i in range(len(probe_names), max_subplots):
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
    
    print(f"Generating visualizations for {len(run_names)} models: {run_names}")
    
    # Generate plots for each model separately
    for run_name in run_names:
        print(f"\nProcessing model: {run_name}")
        
        # Determine output directory based on model type
        if 'gemma' in run_name.lower() or 'mask' in run_name.lower():
            output_dir = main_dir
            print(f"  -> Saving to main/ (Gemma or MASK model)")
        else:
            output_dir = other_dir
            print(f"  -> Saving to other/ (Qwen model)")
        
        # Generate main plots (experiment 2 and 4 best probes) for this model
        for eval_dataset in eval_datasets:
            # Experiment 2 best probes - AUC
            save_path = output_dir / f"experiment_2_best_probes_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_2_best_probes_auc_for_model(eval_dataset, run_name, save_path)
            
            # Experiment 2 best probes - Recall
            save_path = output_dir / f"experiment_2_best_probes_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_2_best_probes_recall_for_model(eval_dataset, run_name, save_path)
            
            # Experiment 4 best probes - AUC
            save_path = output_dir / f"experiment_4_best_probes_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_4_best_probes_auc_for_model(eval_dataset, run_name, save_path)
            
            # Experiment 4 best probes - Recall
            save_path = output_dir / f"experiment_4_best_probes_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_experiment_4_best_probes_recall_for_model(eval_dataset, run_name, save_path)
            
            # LLM upsampling aggregated - AUC
            save_path = output_dir / f"llm_upsampling_aggregated_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_aggregated_auc_for_model(eval_dataset, run_name, save_path)
            
            # LLM upsampling aggregated - Recall
            save_path = output_dir / f"llm_upsampling_aggregated_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_aggregated_recall_for_model(eval_dataset, run_name, save_path)
            
            # Scaling law aggregated - AUC
            save_path = output_dir / f"scaling_law_aggregated_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_scaling_law_aggregated_auc_for_model(eval_dataset, run_name, save_path)
            
            # Scaling law aggregated - Recall
            save_path = output_dir / f"scaling_law_aggregated_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_scaling_law_aggregated_recall_for_model(eval_dataset, run_name, save_path)
    
    print("\nGenerating subplot visualizations...")
    
    # Generate subplot visualizations for each model
    for run_name in run_names:
        print(f"\nProcessing subplots for model: {run_name}")
        
        # Determine output directory based on model type
        if 'gemma' in run_name.lower() or 'mask' in run_name.lower():
            output_dir = main_dir
        else:
            output_dir = other_dir
        
        for eval_dataset in eval_datasets:
            # Experiment 2 subplots
            save_path = output_dir / f"experiment_2_subplots_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_auc_for_model('2-', eval_dataset, run_name, save_path)
            
            # Experiment 2 subplots
            save_path = output_dir / f"experiment_2_subplots_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_recall_for_model('2-', eval_dataset, run_name, save_path)
            
            # Experiment 4 subplots
            save_path = output_dir / f"experiment_4_subplots_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_auc_for_model('4-', eval_dataset, run_name, save_path)
            
            # Experiment 4 subplots
            save_path = output_dir / f"experiment_4_subplots_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_probe_subplots_recall_for_model('4-', eval_dataset, run_name, save_path)
            
            # LLM upsampling subplots
            save_path = output_dir / f"llm_upsampling_subplots_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_subplots_auc_for_model(eval_dataset, run_name, save_path)
            
            # LLM upsampling subplots
            save_path = output_dir / f"llm_upsampling_subplots_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_subplots_recall_for_model(eval_dataset, run_name, save_path)
            
            # Scaling law subplots - AUC
            save_path = output_dir / f"scaling_law_subplots_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_scaling_law_subplots_auc_for_model(eval_dataset, run_name, save_path)
            
            # Scaling law subplots - Recall
            save_path = output_dir / f"scaling_law_subplots_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_scaling_law_subplots_recall_for_model(eval_dataset, run_name, save_path)
    
    print("\nGenerating main aggregated plots...")
    
    # Generate main aggregated plots (all models together)
    for eval_dataset in eval_datasets:
        # LLM upsampling aggregated (Gemma only) - AUC
        save_path = main_dir / f"llm_upsampling_aggregated_auc_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_llm_upsampling_aggregated_auc(eval_dataset, save_path)
        
        # LLM upsampling aggregated (Gemma only) - Recall
        save_path = main_dir / f"llm_upsampling_aggregated_recall_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_llm_upsampling_aggregated_recall(eval_dataset, save_path)
        
        # Scaling law aggregated (Qwen only) - AUC
        save_path = main_dir / f"scaling_law_aggregated_auc_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_aggregated_auc(eval_dataset, save_path)
        
        # Scaling law aggregated (Qwen only) - Recall
        save_path = main_dir / f"scaling_law_aggregated_recall_{eval_dataset}.png"
        if not skip_existing or not save_path.exists():
            plot_scaling_law_aggregated_recall(eval_dataset, save_path)
    
    print("\nGenerating heatmap visualizations...")
    
    for run_name in run_names:
        print(f"\nProcessing heatmaps for model: {run_name}")
        
        # Determine output directory based on model type
        if 'gemma' in run_name.lower() or 'mask' in run_name.lower():
            output_dir = main_dir
        else:
            output_dir = other_dir
        
        for eval_dataset in eval_datasets:
            # Scaling Law Heatmap - AUC
            save_path = output_dir / f"scaling_law_heatmap_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_scaling_law_heatmap_auc_for_model(eval_dataset, run_name, save_path)
            
            # Scaling Law Heatmap - Recall
            save_path = output_dir / f"scaling_law_heatmap_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_scaling_law_heatmap_recall_for_model(eval_dataset, run_name, save_path)
            
            # LLM Upsampling Heatmap - AUC
            save_path = output_dir / f"llm_upsampling_heatmap_auc_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_heatmap_auc_for_model(eval_dataset, run_name, save_path)
            
            # LLM Upsampling Heatmap - Recall
            save_path = output_dir / f"llm_upsampling_heatmap_recall_{eval_dataset}_{run_name}.png"
            if not skip_existing or not save_path.exists():
                plot_llm_upsampling_heatmap_recall_for_model(eval_dataset, run_name, save_path)
    
    print("All visualizations generated successfully!")
    print(f"Main plots (Gemma) saved to: {main_dir}")
    print(f"Other plots (Qwen) saved to: {other_dir}")


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
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax',
            'sae2_last', 'sae2_max', 'sae2_mean', 'sae2_softmax'  # Second SAE variant
        ]
    else:
        # Qwen and other models have 4 SAE probes
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    return probe_names


def get_probe_names_for_subplots(run_name: str = None) -> list:
    """
    Get probe names for subplot visualizations.
    All models now have 12 probes (4 act_sim + 4 linear + 4 SAE).
    """
    # All models now have 12 probes total
    probe_names = [
        'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
        'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
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
            # Gemma SAE (only one variant now)
            return f"{base_label} (Gemma)"
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
                "sae_softmax": {"width": 16384, "l0": 0.408, "label": "SAE (softmax, 16k, L0=0.408)"}
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


def plot_experiment_2_best_probes_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Plot experiment 2 best probes as line chart with confidence intervals for a specific model."""
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
    
    # Filter to only include data from the specified model
    df = df[df['run_name'] == run_name]
    
    if df.empty:
        print(f"No data found for experiment 2, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Debug: Print data before filtering
    print(f"\n=== EXPERIMENT 2 {metric.upper()} PLOT DEBUG for {eval_dataset} - {run_name} ===")
    print(f"Before filtering - Total rows: {len(df)}")
    print(f"Before filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    
    # Debug: Check sample counts
    print(f"Before filtering - num_positive_samples: {df['num_positive_samples'].value_counts().head(10).to_dict()}")
    print(f"Before filtering - num_negative_samples: {df['num_negative_samples'].value_counts().head(10).to_dict()}")
    
    # Apply main plot filters (Gemma SAE topk=1024, linear C=1.0)
    df = apply_main_plot_filters(df)
    
    if df.empty:
        print(f"No data found after applying filters for experiment 2, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Debug: Print data after filtering
    print(f"After filtering - Total rows: {len(df)}")
    print(f"After filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    print(f"After filtering - num_positive_samples: {df['num_positive_samples'].value_counts().to_dict()}")
    print(f"After filtering - num_negative_samples: {df['num_negative_samples'].value_counts().to_dict()}")
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 2, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Debug: Print best probes and their data counts
    print(f"Best probes selected: {best_probes}")
    for probe in best_probes:
        probe_data = df[df['probe_name'] == probe]
        print(f"  {probe}: {len(probe_data)} data rows")
        if not probe_data.empty:
            print(f"    Sample counts: {probe_data['num_positive_samples'].value_counts().to_dict()}")
            print(f"    Seeds: {probe_data['seed'].value_counts().to_dict()}")
    
    print("=== END DEBUG ===\n")
    
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
        
        # Plot line with confidence interval
        ax.plot(x_values, y_means, 'o-', color=colors[i], label=get_probe_label(probe), 
                linewidth=2, markersize=4)
        ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=colors[i])
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for experiment 2 {metric}, eval_dataset={eval_dataset}, run_name={run_name}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # Get number of negative examples for title and format run_name
    num_negative = df['num_negative_samples'].iloc[0] if not df.empty and 'num_negative_samples' in df.columns else None
    
    # Format run_name to be cleaner (e.g., "spam_gemma_9b" -> "Gemma-2-9b")
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
    if num_negative is not None:
        # Convert to int to remove .0
        num_negative_int = int(num_negative)
        ax.set_title(f'{formatted_run_name} Probes On Unbalanced Enron-Spam Dataset\n({num_negative_int} negative examples)')
    else:
        ax.set_title(f'{formatted_run_name} Probes On Unbalanced Enron-Spam Dataset')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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
            return "Qwen-1.5-32B"
        elif '14b' in run_name.lower():
            return "Qwen-1.5-14B"
        elif '8b' in run_name.lower():
            return "Qwen-1.5-8B"
        elif '4b' in run_name.lower():
            return "Qwen-1.5-4B"
        elif '1.7b' in run_name.lower():
            return "Qwen-1.5-1.7B"
        elif '0.6b' in run_name.lower():
            return "Qwen-1.5-0.6B"
        else:
            return "Qwen-1.5"
    elif 'mask' in run_name.lower():
        return "MASK"
    else:
        return run_name


def format_dataset_name_for_display(eval_dataset: str) -> str:
    """Format eval_dataset for display in plot titles."""
    if 'enron' in eval_dataset.lower() or 'spam' in eval_dataset.lower():
        return "Enron-Spam"
    else:
        # Capitalize first letter and replace underscores with spaces
        return eval_dataset.replace('_', ' ').title()


def plot_experiment_2_best_probes_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot experiment 2 best probes AUC as line chart with confidence intervals for a specific model."""
    plot_experiment_2_best_probes_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_experiment_2_best_probes_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot experiment 2 best probes Recall @ FPR=0.01 as line chart with confidence intervals for a specific model."""
    plot_experiment_2_best_probes_for_model(eval_dataset, run_name, save_path, metric='recall')


def plot_experiment_4_best_probes_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Plot experiment 4 best probes as line chart with confidence intervals for a specific model."""
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
    
    # Filter to only include data from the specified model
    df = df[df['run_name'] == run_name]
    
    if df.empty:
        print(f"No data found for experiment 4, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Debug: Print data before filtering
    print(f"\n=== EXPERIMENT 4 {metric.upper()} PLOT DEBUG for {eval_dataset} - {run_name} ===")
    print(f"Before filtering - Total rows: {len(df)}")
    print(f"Before filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    
    # Debug: Check sample counts
    print(f"Before filtering - num_positive_samples: {df['num_positive_samples'].value_counts().head(10).to_dict()}")
    print(f"Before filtering - num_negative_samples: {df['num_negative_samples'].value_counts().head(10).to_dict()}")
    
    # Apply main plot filters (Gemma SAE topk=1024, linear C=1.0)
    df = apply_main_plot_filters(df)
    
    if df.empty:
        print(f"No data found after applying filters for experiment 4, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Debug: Print data after filtering
    print(f"After filtering - Total rows: {len(df)}")
    print(f"After filtering - Probe types: {df['probe_name'].value_counts().to_dict()}")
    print(f"After filtering - num_positive_samples: {df['num_positive_samples'].value_counts().to_dict()}")
    print(f"After filtering - num_negative_samples: {df['num_negative_samples'].value_counts().to_dict()}")
    
    # Get best probes from each category
    best_probes = get_best_probes_by_category(df)
    
    if not best_probes:
        print(f"No valid probes found for experiment 4, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Debug: Print best probes and their data counts
    print(f"Best probes selected: {best_probes}")
    for probe in best_probes:
        probe_data = df[df['probe_name'] == probe]
        print(f"  {probe}: {len(probe_data)} data rows")
        if not probe_data.empty:
            print(f"    Sample counts: {probe_data['num_positive_samples'].value_counts().to_dict()}")
            print(f"    Seeds: {probe_data['seed'].value_counts().to_dict()}")
    
    print("=== END DEBUG ===\n")
    
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
        
        # Plot line with confidence interval
        ax.plot(x_values, y_means, 'o-', color=colors[i], label=get_probe_label(probe), 
                linewidth=2, markersize=4)
        ax.fill_between(x_values, y_lower, y_upper, alpha=0.3, color=colors[i])
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for experiment 4 {metric}, eval_dataset={eval_dataset}, run_name={run_name}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # Format run_name and eval_dataset for cleaner title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'AUC'
    elif metric == 'recall':
        ylabel = 'Recall @ FPR=0.01'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_xlabel('Number of training examples per class\n(x negative, x positive)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{formatted_run_name} Probes on Balanced Enron-Spam Dataset\n(equal positive and negative samples)')
    
    # Set log scale only if all x values are positive
    all_x_values = []
    for probe in best_probes:
        probe_data = df[df['probe_name'] == probe]
        if not probe_data.empty:
            all_x_values.extend(probe_data['num_positive_samples'].unique())
    
    if all_x_values and min(all_x_values) > 0:
        ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_4_best_probes_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot experiment 4 best probes AUC as line chart with confidence intervals for a specific model."""
    plot_experiment_4_best_probes_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_experiment_4_best_probes_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot experiment 4 best probes Recall @ FPR=0.01 as line chart with confidence intervals for a specific model."""
    plot_experiment_4_best_probes_for_model(eval_dataset, run_name, save_path, metric='recall')


# Placeholder functions for other plot types (to be implemented later)
def plot_llm_upsampling_aggregated_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Plot LLM upsampling with median performance across all probes for Gemma models only."""
    setup_plot_style()
    
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
    fig, ax = plt.subplots(figsize=(8, 6))
    
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
                label=f'{ratio_int}x', linewidth=2, markersize=4)
        lines_plotted += 1
    
    # Don't save if no lines were plotted
    if lines_plotted == 0:
        print(f"No data to plot for LLM upsampling {metric}, eval_dataset={eval_dataset}, run_name={run_name}")
        plt.close()
        return
    
    # Set proper ylim based on actual data
    if all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.02)  # Reduced from 0.05 to 0.02 to cut off higher
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
        title_suffix = 'AUC'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01 (across all probes)'
        title_suffix = 'Recall'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    # Format run_name and eval_dataset for cleaner title
    formatted_run_name = format_run_name_for_display(run_name)
    formatted_dataset = format_dataset_name_for_display(eval_dataset)
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{formatted_run_name} Probes on LLM-Upsampled {formatted_dataset}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Only add legend if there are lines plotted
    if all_y_values:
        ax.legend(title='Upsampling \nFactor')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_aggregated_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot LLM upsampling with median AUC performance across all probes for Gemma models only."""
    plot_llm_upsampling_aggregated_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_llm_upsampling_aggregated_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot LLM upsampling with median Recall performance across all probes for Gemma models only."""
    plot_llm_upsampling_aggregated_for_model(eval_dataset, run_name, save_path, metric='recall')


def plot_scaling_law_aggregated_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Plot scaling law with median performance across all probes for Qwen models only."""
    setup_plot_style()
    
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
    fig, ax = plt.subplots(figsize=(8, 6))
    
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
    
    # Plot line
    ax.plot(x_values, y_medians, 'o-', color='blue', 
            label=f'{model_size}B', linewidth=2, markersize=4)
    
    # Set proper ylim based on actual data
    if y_medians:
        y_min = max(0.0, min(y_medians) - 0.05)
        y_max = min(1.0, max(y_medians) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    # Set metric-specific parameters
    if metric == 'auc':
        ylabel = 'Median AUC (across all probes)'
        title_suffix = 'AUC'
    elif metric == 'recall':
        ylabel = 'Median Recall @ FPR=0.01 (across all probes)'
        title_suffix = 'Recall'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Scaling Law: Median {title_suffix} Performance\n{run_name} - {eval_dataset}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_aggregated_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot scaling law with median AUC performance across all probes for Qwen models only."""
    plot_scaling_law_aggregated_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_scaling_law_aggregated_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Plot scaling law with median Recall performance across all probes for Qwen models only."""
    plot_scaling_law_aggregated_for_model(eval_dataset, run_name, save_path, metric='recall')


def plot_probe_subplots_auc_for_model(experiment: str, eval_dataset: str, run_name: str, save_path: Path):
    """Create subplot visualization for all probes AUC for a specific model."""
    plot_probe_subplots_unified_for_model(experiment, eval_dataset, run_name, save_path, metric='auc')


def plot_probe_subplots_recall_for_model(experiment: str, eval_dataset: str, run_name: str, save_path: Path):
    """Create subplot visualization for all probes Recall for a specific model."""
    plot_probe_subplots_unified_for_model(experiment, eval_dataset, run_name, save_path, metric='recall')


def plot_probe_subplots_unified_for_model(experiment: str, eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create subplot visualization for all probes for a specific model - unified function for AUC and Recall."""
    setup_plot_style()
    
    # Get data for the experiment for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment=experiment,
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for experiment {experiment}, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Define the probe names we want to include
    # For subplots, we need to determine if this is a Gemma experiment
    if 'gemma' in run_name.lower():
        # Gemma has 12 probes total (4 act_sim + 4 linear + 4 SAE)
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    else:
        # Qwen and other models have 12 probes total (4 act_sim + 4 linear + 4 SAE)
        probe_names = [
            'act_sim_last', 'act_sim_max', 'act_sim_mean', 'act_sim_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
            'sae_last', 'sae_max', 'sae_mean', 'sae_softmax'
        ]
    
    # Create subplot grid based on number of probes
    # All models now have 12 probes (4 act_sim + 4 linear + 4 SAE)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
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
    
    for i, probe in enumerate(probe_names):  # Use all probes
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
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
            ax.set_xlabel('Training Size')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
    
    # Set consistent ylim across all subplots
    if all_y_values:
        y_min = max(y_min_default, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    
    # Set experiment-specific title
    if experiment == '2-':
        experiment_title = 'Unbalanced'
    elif experiment == '4-':
        experiment_title = 'Balanced'
    else:
        experiment_title = f'Experiment {experiment}'
    
    fig.suptitle(f'{run_name} - {experiment_title} {title_suffix}\n{eval_dataset}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_subplots_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create LLM upsampling subplot visualization for all probes AUC for a specific model."""
    plot_llm_upsampling_subplots_unified_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_llm_upsampling_subplots_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create LLM upsampling subplot visualization for all probes Recall for a specific model."""
    plot_llm_upsampling_subplots_unified_for_model(eval_dataset, run_name, save_path, metric='recall')


def plot_llm_upsampling_subplots_unified_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create LLM upsampling subplot visualization for all probes for a specific model - unified function for AUC and Recall."""
    setup_plot_style()
    
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
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
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
                    
                    ax.plot(x_values, y_means, 'o-', color=color, markersize=3, linewidth=1, alpha=0.7)
            
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
            ax.set_xlabel('Training Size')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
    
    # Set consistent ylim across all subplots
    if all_y_values:
        y_min = max(y_min_default, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    
    fig.suptitle(f'{run_name} - LLM Upsampling {title_suffix}\n{eval_dataset}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_subplots_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create scaling law subplot visualization for all probes AUC for a specific model."""
    plot_scaling_law_subplots_unified_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_scaling_law_subplots_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create scaling law subplot visualization for all probes Recall for a specific model."""
    plot_scaling_law_subplots_unified_for_model(eval_dataset, run_name, save_path, metric='recall')


def plot_scaling_law_subplots_unified_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create scaling law subplot visualization for all probes for a specific model - unified function for AUC and Recall."""
    setup_plot_style()
    
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
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
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
            
            ax.plot(x_values, y_means, 'o-', color=color, markersize=4, linewidth=1.5)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
            ax.set_xlabel('Training Size')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(wrap_text(get_detailed_probe_label(probe), 25))
    
    # Set consistent ylim across all subplots
    if all_y_values:
        y_min = max(y_min_default, min(all_y_values) - 0.05)
        y_max = min(1.0, max(all_y_values) + 0.05)
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    
    fig.suptitle(f'{run_name} - Scaling Law {title_suffix}\n{eval_dataset}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_heatmap_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create scaling law heatmap for a specific model."""
    setup_plot_style()
    
    # Only generate for Qwen models
    if 'qwen' not in run_name.lower():
        print(f"Skipping scaling law heatmap for {run_name} - only for Qwen models")
        return
    
    # Get data for experiment 2 for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='2-',
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for scaling law heatmap, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create heatmap data
    pivot_data = df.pivot_table(
        values=metric,
        index='probe_name',
        columns='num_positive_samples',
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        print(f"No data to plot for scaling law heatmap, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    
    # Set metric-specific parameters
    if metric == 'auc':
        title_suffix = 'AUC'
    elif metric == 'recall':
        title_suffix = 'Recall @ FPR=0.01'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_title(f'Scaling Law Heatmap: {title_suffix}\n{run_name} - {eval_dataset}')
    ax.set_xlabel('Number of positive examples in the train set')
    ax.set_ylabel('Probe Name')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_law_heatmap_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create scaling law heatmap for AUC for a specific model."""
    plot_scaling_law_heatmap_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_scaling_law_heatmap_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create scaling law heatmap for Recall for a specific model."""
    plot_scaling_law_heatmap_for_model(eval_dataset, run_name, save_path, metric='recall')


def plot_llm_upsampling_heatmap_for_model(eval_dataset: str, run_name: str, save_path: Path, metric: str = 'auc'):
    """Create LLM upsampling heatmap for a specific model."""
    setup_plot_style()
    
    # Only generate for Gemma models
    if 'gemma' not in run_name.lower():
        print(f"Skipping LLM upsampling heatmap for {run_name} - only for Gemma models")
        return
    
    # Get data for experiment 3 for this specific model
    df = get_data_for_visualization(
        eval_dataset=eval_dataset,
        experiment='3-',
        run_name=run_name,
        exclude_attention=True
    )
    
    if df.empty:
        print(f"No data found for LLM upsampling heatmap, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create heatmap data
    pivot_data = df.pivot_table(
        values=metric,
        index='probe_name',
        columns=['num_positive_samples', 'llm_upsampling_ratio'],
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        print(f"No data to plot for LLM upsampling heatmap, eval_dataset={eval_dataset}, run_name={run_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    
    # Set metric-specific parameters
    if metric == 'auc':
        title_suffix = 'AUC'
    elif metric == 'recall':
        title_suffix = 'Recall @ FPR=0.01'
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'auc' or 'recall'.")
    
    ax.set_title(f'LLM Upsampling Heatmap: {title_suffix}\n{run_name} - {eval_dataset}')
    ax.set_xlabel('(Number of positive examples, Upsampling ratio)')
    ax.set_ylabel('Probe Name')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_llm_upsampling_heatmap_auc_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create LLM upsampling heatmap for AUC for a specific model."""
    plot_llm_upsampling_heatmap_for_model(eval_dataset, run_name, save_path, metric='auc')


def plot_llm_upsampling_heatmap_recall_for_model(eval_dataset: str, run_name: str, save_path: Path):
    """Create LLM upsampling heatmap for Recall for a specific model."""
    plot_llm_upsampling_heatmap_for_model(eval_dataset, run_name, save_path, metric='recall')


if __name__ == "__main__":
    generate_all_visualizations()
