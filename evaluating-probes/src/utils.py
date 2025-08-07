
import numpy as np
import json
from typing import List, Any
from src.logger import Logger
from pathlib import Path
from dataclasses import asdict
from src.probes import LinearProbe, AttentionProbe, MassMeanProbe, ActivationSimilarityProbe, SAEProbe, BaseProbeNonTrainable
from configs.probes import PROBE_CONFIGS
from src.data import Dataset
import pandas as pd
import os 

def should_skip_dataset(dataset_name, data=None, logger=None):
    """This defines conditions for datasets we should always skip. Now supports passing just a dataset_name (loads from cleaned/)."""
    # If data is None or a string, try to load the dataset from cleaned folder
    if data is None or isinstance(data, str):
        cleaned_dir = Path(__file__).parent.parent / "datasets/cleaned"
        # Try to find the file by name or by prefix
        candidates = list(cleaned_dir.glob(f"*{dataset_name}*.csv"))
        if not candidates:
            if logger:
                logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Could not find CSV in {cleaned_dir}.")
            return True
        csv_path = candidates[0]
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            if logger:
                logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Could not load CSV: {e}")
            return True
        # Try to infer task type
        if 'target' not in df.columns:
            if logger:
                logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': No 'target' column in CSV.")
            return True
        y = df['target']
        # Try to infer if classification or regression
        if pd.api.types.is_numeric_dtype(y):
            # Heuristic: if all values are floats and not just 0/1, treat as regression
            unique_vals = y.unique()
            if len(unique_vals) > 2 or any(isinstance(v, float) and not v.is_integer() for v in unique_vals):
                if logger:
                    logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Continuous/regression data is not supported.")
                return True
        # Check for binary classification
        unique_classes = pd.Series(y).unique()
        if len(unique_classes) != 2:
            if logger:
                logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Expected binary classification, got {len(unique_classes)} classes.")
            return True
        # Check for minimum samples per class
        counts = pd.Series(y).value_counts()
        if counts.min() < 2:
            if logger:
                logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': At least one class has <2 samples (counts: {dict(counts)}).")
            return True
        return False
    return False


def dump_loss_history(losses: List[float], out_path: Path, logger: Logger | None = None):
     """
     Write `[loss0, loss1, ...]` as pretty-printed JSON.
     """
     with open(out_path, "w") as fp:
         json.dump(losses, fp, indent=2)
     if logger:
         logger.log(f"  - 😋 Loss history saved in {out_path.name}")


def extract_aggregation_from_config(config_name: str, architecture_name: str) -> str:
    """Extract aggregation from config for backward compatibility."""
    config = PROBE_CONFIGS[config_name]
    if hasattr(config, 'aggregation'): # act sim and linear have aggregation in config
        return config.aggregation
    elif architecture_name == "attention":
        return "attention"
    elif architecture_name in ["mass_mean"]:  # mass_mean_iid removed due to numerical instability
        return "mass_mean"  # Mass-mean probes don't use aggregation
    else:
        return "mean"  # Default fallback


def get_probe_architecture(architecture_name: str, d_model: int, device, config: dict):
    """Create probe architecture with filtered config parameters."""
    if architecture_name == "linear":
        return LinearProbe(d_model=d_model, device=device, **config)
    if architecture_name == "attention":
        return AttentionProbe(d_model=d_model, device=device, **config)
    if architecture_name.startswith("sae"):
        return SAEProbe(d_model=d_model, device=device, **config)
    if architecture_name in ["mass_mean"]:  # mass_mean_iid removed due to numerical instability
        # Mass-mean probes need use_iid parameter from config
        return MassMeanProbe(d_model=d_model, device=device, **config)
    if architecture_name.startswith("act_sim"):
        # Activation similarity probes - filter out batch_size from constructor args
        filtered_config = {k: v for k, v in config.items() if k != 'batch_size'}
        return ActivationSimilarityProbe(d_model=d_model, device=device, **filtered_config)
    raise ValueError(f"Unknown architecture: {architecture_name}")


def get_probe_filename_prefix(train_ds, arch_name, layer, component, config_name, contrast_fn=None):
    # Should ideally just use arch_name and not agg_name but now we need to be backward compatible
    agg_name = extract_aggregation_from_config(config_name, arch_name)
    
    base_prefix = f"train_on_{train_ds}_{arch_name}_{agg_name}_L{layer}_{component}"
    if contrast_fn is not None:
        # Add contrast function name to filename to distinguish from regular probes
        contrast_name = contrast_fn.__name__ if hasattr(contrast_fn, '__name__') else 'contrast'
        base_prefix += f"_{contrast_name}"
    return base_prefix


def rebuild_suffix(rebuild_config):
    if not rebuild_config:
        return "original"
    
    # Handle new LLM upsampling format
    if 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
        n_real_neg = rebuild_config.get('n_real_neg')
        n_real_pos = rebuild_config.get('n_real_pos')
        upsampling_factor = rebuild_config.get('upsampling_factor')
        return f"llm_neg{n_real_neg}_pos{n_real_pos}_{upsampling_factor}x"
    
    # Handle original rebuild_config formats
    if 'class_counts' in rebuild_config:
        cc = rebuild_config['class_counts']
        cc_str = '_'.join([f"class{cls}_{cc[cls]}" for cls in sorted(cc)])
        return f"{cc_str}"
    elif 'class_percents' in rebuild_config:
        cp = rebuild_config['class_percents']
        cp_str = '_'.join([f"class{cls}_{int(cp[cls]*100)}pct" for cls in sorted(cp)])
        return f"{cp_str}_total{rebuild_config['total_samples']}"
    else:
        return "custom"


def resample_params_to_str(params):
    if params is None:
        return "original"
    
    # Handle new LLM upsampling format
    if 'llm_upsampling' in params and params['llm_upsampling']:
        n_real_neg = params.get('n_real_neg')
        n_real_pos = params.get('n_real_pos')
        upsampling_factor = params.get('upsampling_factor')
        return f"llm_neg{n_real_neg}_pos{n_real_pos}_{upsampling_factor}x"
    
    # Handle original rebuild_config formats
    if 'class_counts' in params:
        cc = params['class_counts']
        cc_str = '_'.join([f"class{cls}_{cc[cls]}" for cls in sorted(cc)])
        return f"{cc_str}"
    elif 'class_percents' in params:
        cp = params['class_percents']
        cp_str = '_'.join([f"class{cls}_{int(cp[cls]*100)}pct" for cls in sorted(cp)])
        return f"{cp_str}_total{params['total_samples']}"
    else:
        return "custom"


def get_dataset(name, model, device, seed):
    return Dataset(name, model=model, device=device, seed=seed)

def get_effective_seeds(config):
    """Extract seeds from config, supporting both single seed and multiple seeds.
    Returns a list of seeds for uniform processing.
    """
    if 'seeds' in config:
        # Multiple seeds specified
        seeds = config['seeds']
        if isinstance(seeds, list):
            return seeds
        else:
            # Handle case where seeds might be a single value
            return [seeds]
    elif 'seed' in config:
        # Single seed specified (backward compatibility)
        return [config['seed']]
    else:
        # Default seed if none specified
        return [42]

def get_effective_seed_for_rebuild_config(global_seed, rebuild_config):
    """
    For LLM upsampling experiments, rebuild_config seed overrides global seed.
    For other experiments, use global seed.
    """
    if rebuild_config and 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
        # LLM upsampling: rebuild_config seed takes precedence
        return rebuild_config.get('seed', global_seed)
    else:
        # Regular experiments: global seed takes precedence
        return global_seed

def generate_llm_upsampling_configs(n_real_neg, n_real_pos_list, upsampling_factors, seed):
    """
    Generate rebuild_configs for LLM upsampling experiments.
    Creates a config for each combination of n_real_pos and upsampling_factor.
    
    Args:
        n_real_neg: Fixed number of real negative samples to use
        n_real_pos_list: List of real positive sample counts to try (e.g., [1, 2, 3, 4, 5])
        upsampling_factors: List of upsampling factors to try (e.g., [1, 2, 3, 4, 5])
        seed: Random seed for the experiments
    """
    configs = []
    for n_real_pos in n_real_pos_list:
        for factor in upsampling_factors:
            configs.append({
                'llm_upsampling': True,
                'n_real_neg': n_real_neg,
                'n_real_pos': n_real_pos,
                'upsampling_factor': factor,
                'seed': seed  # Override global seed for LLM experiments
            })
    return configs
