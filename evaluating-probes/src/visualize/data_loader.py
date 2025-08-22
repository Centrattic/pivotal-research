import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


def load_metrics_data() -> pd.DataFrame:
    """Load the metrics index CSV file."""
    csv_path = Path("src/visualize/metrics_index.csv")
    return pd.read_csv(csv_path)


def extract_info_from_filename(filename: str) -> Dict[str, str]:
    """Extract experiment info from filename."""
    info = {}
    
    # Extract run name (e.g., spam_gemma_9b, spam_qwen_0.6b)
    run_match = re.search(r'results/([^/]+)/', filename)
    if run_match:
        info['run_name'] = run_match.group(1)
    
    # Extract seed
    seed_match = re.search(r'seed_(\d+)', filename)
    if seed_match:
        info['seed'] = seed_match.group(1)
    
    # Extract experiment type (1-spam-pred-auc, 2-*, 3-*, 4-*)
    exp_match = re.search(r'seed_\d+/([^/]+)/', filename)
    if exp_match:
        info['experiment'] = exp_match.group(1)
    
    # Extract eval dataset
    eval_match = re.search(r'eval_on_([^_]+(?:_[^_]+)*?)__', filename)
    if eval_match:
        info['eval_dataset'] = eval_match.group(1)
    
    # Extract train dataset
    train_match = re.search(r'train_on_([^_]+(?:_[^_]+)*?)_', filename)
    if train_match:
        info['train_dataset'] = train_match.group(1)
    
    # Extract training sample counts from class0_X_class1_Y pattern
    class_match = re.search(r'class0_(\d+)_class1_(\d+)', filename)
    if class_match:
        info['num_negative_samples'] = int(class_match.group(1))
        info['num_positive_samples'] = int(class_match.group(2))
    else:
        # For LLM upsampling experiments, look for llm_negX_posY pattern
        llm_match = re.search(r'llm_neg(\d+)_pos(\d+)', filename)
        if llm_match:
            info['num_negative_samples'] = int(llm_match.group(1))
            info['num_positive_samples'] = int(llm_match.group(2))
        else:
            # Also check for posX pattern in LLM upsampling (matching old code)
            pos_match = re.search(r'pos(\d+)_', filename)
            if pos_match:
                info['num_positive_samples'] = int(pos_match.group(1))
                info['num_negative_samples'] = None  # Not specified in this pattern
            else:
                # For attention probes, look for different patterns
                # Check if this is an attention probe filename
                if 'attention' in filename:
                    # For attention probes, we need to look for different patterns
                    # Let's check for patterns like class0_X_class1_Y in the filename
                    # This should work for both old and new attention probe patterns
                    attention_class_match = re.search(r'class0_(\d+)_class1_(\d+)', filename)
                    if attention_class_match:
                        info['num_negative_samples'] = int(attention_class_match.group(1))
                        info['num_positive_samples'] = int(attention_class_match.group(2))
                    else:
                        # If no pattern found, set to None
                        info['num_negative_samples'] = None
                        info['num_positive_samples'] = None
                else:
                    info['num_negative_samples'] = None
                    info['num_positive_samples'] = None
    
    # Extract LLM upsampling ratio (for experiment 3)
    # Look for pattern like pos1_10x or pos2_20x (matching old code)
    upsampling_match = re.search(r'pos(\d+)_([1-9]\d*)x', filename)
    if upsampling_match:
        info['llm_upsampling_ratio'] = int(upsampling_match.group(2))
    else:
        # Fallback to simpler pattern
        upsampling_match = re.search(r'_(\d+)x_', filename)
        if upsampling_match:
            info['llm_upsampling_ratio'] = int(upsampling_match.group(1))
        else:
            info['llm_upsampling_ratio'] = None
    
    # Extract Qwen model size for scaling analysis
    qwen_size_match = re.search(r'qwen_([0-9.]+)b', filename, re.IGNORECASE)
    if qwen_size_match:
        info['qwen_model_size'] = float(qwen_size_match.group(1))
    else:
        info['qwen_model_size'] = None
    
    # Extract probe name and determine architecture type
    probe_architecture = None
    if 'act_sim' in filename:
        probe_architecture = 'act_sim'
        if 'act_sim_last' in filename:
            info['probe_name'] = 'act_sim_last'
        elif 'act_sim_max' in filename:
            info['probe_name'] = 'act_sim_max'
        elif 'act_sim_mean' in filename:
            info['probe_name'] = 'act_sim_mean'
        elif 'act_sim_softmax' in filename:
            info['probe_name'] = 'act_sim_softmax'
    elif 'sae' in filename:
        probe_architecture = 'sae'
        # For SAE probes, distinguish variants and map to simplified names
        is_gemma = 'gemma' in filename.lower()
        is_262k = ('sae_262k' in filename) or ('sae2_' in filename) or ('262k' in filename)
        if is_gemma and is_262k:
            # Exclude Gemma's 262k SAE variant from simplified SAE set
            probe_architecture = None
            info['probe_name'] = None
        else:
            # Map based on aggregation suffix present in filename
            if re.search(r'sae.*_last_', filename):
                info['probe_name'] = 'sae_last'
            elif re.search(r'sae.*_max_', filename):
                info['probe_name'] = 'sae_max'
            elif re.search(r'sae.*_mean_', filename):
                info['probe_name'] = 'sae_mean'
            elif re.search(r'sae.*_softmax_', filename):
                info['probe_name'] = 'sae_softmax'
            else:
                # If SAE but aggregation not detected, leave unset
                info['probe_name'] = None
    elif 'sklearn_linear' in filename:
        probe_architecture = 'sklearn_linear'
        if 'sklearn_linear_last' in filename:
            info['probe_name'] = 'sklearn_linear_last'
        elif 'sklearn_linear_max' in filename:
            info['probe_name'] = 'sklearn_linear_max'
        elif 'sklearn_linear_mean' in filename:
            info['probe_name'] = 'sklearn_linear_mean'
        elif 'sklearn_linear_softmax' in filename:
            info['probe_name'] = 'sklearn_linear_softmax'
    elif 'attention' in filename:
        probe_architecture = 'attention'
        info['probe_name'] = 'attention'
    
    # Extract hyperparameters based on architecture type
    # Default values according to the code
    default_C = 1.0
    default_weight_decay = 0.0
    default_topk = 3584
    default_lr = 5e-4
    
    # Extract C value (for sklearn_linear and sae)
    if probe_architecture in ['sklearn_linear', 'sae']:
        c_match = re.search(r'C([0-9eE\.+-]+)', filename)
        if c_match:
            try:
                info['C'] = float(c_match.group(1))
            except ValueError:
                info['C'] = default_C
        else:
            info['C'] = default_C
    
    # Extract top_k value (for sae)
    if probe_architecture == 'sae':
        topk_match = re.search(r'topk(\d+)', filename)
        if topk_match:
            try:
                info['topk'] = int(topk_match.group(1))
            except ValueError:
                info['topk'] = default_topk
        else:
            info['topk'] = default_topk
    
    # Extract learning rate and weight decay (for attention)
    if probe_architecture == 'attention':
        lr_match = re.search(r'lr([0-9eE\.+-]+)', filename)
        if lr_match:
            try:
                info['lr'] = float(lr_match.group(1))
            except ValueError:
                info['lr'] = default_lr
        else:
            info['lr'] = default_lr
        
        wd_match = re.search(r'wd([0-9eE\.+-]+)', filename)
        if wd_match:
            try:
                info['weight_decay'] = float(wd_match.group(1))
            except ValueError:
                info['weight_decay'] = default_weight_decay
        else:
            info['weight_decay'] = default_weight_decay
    
    # For act_sim, no hyperparameters are needed (non-trainable)
    if probe_architecture == 'act_sim':
        info['C'] = None
        info['topk'] = None
        info['lr'] = None
        info['weight_decay'] = None
    
    return info


def get_data_for_visualization(
    eval_dataset: Optional[str] = None,
    probe_name: Optional[str] = None,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    seeds: Optional[List[str]] = None,
    train_dataset: Optional[str] = None,
    exclude_attention: bool = False
) -> pd.DataFrame:
    """
    Retrieve data from metrics_index.csv based on specified parameters.
    
    Args:
        eval_dataset: Evaluation dataset to filter by
        probe_name: Probe name to filter by (can be partial for SAE probes)
        experiment: Experiment type (e.g., '2-*', '4-*')
        run_name: Run name to filter by (e.g., 'spam_gemma_9b')
        seeds: List of seeds to include
        train_dataset: Training dataset to filter by
        exclude_attention: Whether to exclude attention probes
    
    Returns:
        DataFrame with filtered data and extracted metadata columns
    """
    df = load_metrics_data()
    
    # Extract metadata from filenames
    metadata_list = []
    for filename in df['filename']:
        metadata = extract_info_from_filename(filename)
        metadata_list.append(metadata)
    
    # Add metadata columns
    for key in ['run_name', 'seed', 'experiment', 'eval_dataset', 'train_dataset', 'probe_name']:
        df[key] = [meta.get(key) for meta in metadata_list]
    
    # Add hyperparameter columns
    for key in ['C', 'topk', 'lr', 'weight_decay']:
        df[key] = [meta.get(key) for meta in metadata_list]
    
    # Add additional metadata columns
    for key in ['num_negative_samples', 'num_positive_samples', 'llm_upsampling_ratio', 'qwen_model_size']:
        df[key] = [meta.get(key) for meta in metadata_list]
    
    # Apply filters
    if eval_dataset is not None:
        df = df[df['eval_dataset'] == eval_dataset]
    
    if probe_name is not None:
        if 'sae' in probe_name:
            # For SAE probes, match the specific probe name
            # All SAE variants now use the same naming (no more sae2_ distinction)
            # But we need to explicitly exclude 262k SAEs
            if 'last' in probe_name:
                df = df[df['filename'].str.contains('sae.*_last_') & ~df['filename'].str.contains('sae_262k')]
            elif 'max' in probe_name:
                df = df[df['filename'].str.contains('sae.*_max_') & ~df['filename'].str.contains('sae_262k')]
            elif 'mean' in probe_name:
                df = df[df['filename'].str.contains('sae.*_mean_') & ~df['filename'].str.contains('sae_262k')]
            elif 'softmax' in probe_name:
                df = df[df['filename'].str.contains('sae.*_softmax_') & ~df['filename'].str.contains('sae_262k')]
        else:
            df = df[df['filename'].str.contains(probe_name)]
    
    if experiment is not None:
        # Convert experiment column to string and handle NaN values
        df = df[df['experiment'].astype(str).str.startswith(experiment)]
    
    if run_name is not None:
        df = df[df['run_name'] == run_name]
    
    if seeds is not None:
        df = df[df['seed'].isin(seeds)]
    
    if train_dataset is not None:
        df = df[df['train_dataset'] == train_dataset]
    
    if exclude_attention:
        df = df[~df['filename'].str.contains('attention')]
    
    return df.reset_index(drop=True)


def get_probe_names() -> List[str]:
    """Get list of all available probe names (excluding attention and 262k SAEs)."""
    df = load_metrics_data()
    
    # Extract probe names from all files
    probe_names = set()
    for filename in df['filename']:
        metadata = extract_info_from_filename(filename)
        if metadata.get('probe_name') and 'attention' not in metadata['probe_name']:
            probe_names.add(metadata['probe_name'])
    
    return sorted(list(probe_names))


def get_eval_datasets() -> List[str]:
    """Get list of all available evaluation datasets."""
    df = load_metrics_data()
    
    # Extract eval datasets from all files
    eval_datasets = set()
    for filename in df['filename']:
        metadata = extract_info_from_filename(filename)
        if metadata.get('eval_dataset'):
            eval_datasets.add(metadata['eval_dataset'])
    
    return sorted(list(eval_datasets))


def get_run_names() -> List[str]:
    """Get list of all available run names."""
    df = load_metrics_data()
    
    # Extract run names from all files
    run_names = set()
    for filename in df['filename']:
        metadata = extract_info_from_filename(filename)
        if metadata.get('run_name'):
            run_names.add(metadata['run_name'])
    
    return sorted(list(run_names))


def get_experiment_types() -> List[str]:
    """Get list of all available experiment types."""
    df = load_metrics_data()
    
    # Extract experiment types from all files
    experiment_types = set()
    for filename in df['filename']:
        metadata = extract_info_from_filename(filename)
        if metadata.get('experiment'):
            experiment_types.add(metadata['experiment'])
    
    return sorted(list(experiment_types))


def filter_gemma_sae_topk_1024(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include Gemma SAEs with topk=1024 (exclude 262k SAEs)."""
    # Filter for Gemma runs
    gemma_df = df[df['run_name'].str.contains('gemma', case=False, na=False)]
    
    # Filter for SAE probes with topk=1024 and explicitly exclude 262k SAEs
    # We want only the 16k SAEs, not the 262k SAEs
    filtered_df = gemma_df[
        (gemma_df['probe_name'].str.contains('sae', na=False)) & 
        (gemma_df['topk'] == 1024) &
        (~gemma_df['filename'].str.contains('sae_262k', na=False)) &  # Exclude 262k SAEs
        (~gemma_df['filename'].str.contains('sae2_', na=False)) &     # Exclude sae2_ (262k variant)
        (~gemma_df['filename'].str.contains('262k', na=False))        # Exclude any 262k references
    ]
    
    # Fallback: if nothing left after strict filter, relax to include any non-262k SAEs
    if filtered_df.empty and not gemma_df.empty:
        relaxed = gemma_df[
            (gemma_df['probe_name'].str.contains('sae', na=False)) &
            (~gemma_df['filename'].str.contains('sae_262k', na=False)) &
            (~gemma_df['filename'].str.contains('sae2_', na=False)) &
            (~gemma_df['filename'].str.contains('262k', na=False))
        ]
        if not relaxed.empty:
            print("[filters] No Gemma SAE rows with topk=1024; falling back to all non-262k Gemma SAEs")
            return relaxed
    
    return filtered_df


def filter_linear_probes_c_1_0(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include linear probes with C=1.0 (or no C specified)."""
    # Filter for linear probes
    linear_df = df[df['probe_name'].str.contains('sklearn_linear', na=False)]
    
    # Filter for C=1.0 or no C specified (which defaults to 1.0)
    # Check for C=1.0, C=1e0, or no C in filename
    filtered_df = linear_df[
        (linear_df['C'] == 1.0) | 
        (linear_df['C'].isna()) |  # No C specified, defaults to 1.0
        (linear_df['filename'].str.contains('C1e0', na=False))  # C=1e0 format
    ]
    
    return filtered_df


def filter_default_attention_probes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include default attention probes (no wd_/lr_ parameters OR specific default values)."""
    # Filter for attention probes
    attention_df = df[df['probe_name'].str.contains('attention', na=False)]
    
    # Filter for default attention probes:
    # 1. No wd_ and lr_ parameters in filename, OR
    # 2. Specific default values: lr_6p31e-04_wd_0p00e00
    filtered_df = attention_df[
        (
            (~attention_df['filename'].str.contains('wd_', na=False)) &  # No weight decay parameter
            (~attention_df['filename'].str.contains('lr_', na=False))    # No learning rate parameter
        ) |
        (attention_df['filename'].str.contains('lr_6p31e-04_wd_0p00e00', na=False))  # Specific default values
    ]
    
    return filtered_df



