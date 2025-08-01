
import numpy as np
import json
from typing import List, Any
from src.logger import Logger
from pathlib import Path
from dataclasses import asdict
from src.probes import LinearProbe, AttentionProbe, MassMeanProbe, ActivationSimilarityProbe, SAEProbe, BaseProbeNonTrainable
from configs.probes import PROBE_CONFIGS
from src.data import Dataset

def should_skip_dataset(dataset_name, data, logger=None):
    """ This defines conditions for datasets we should always skip. We have separate conditions for skipping a dataset for evaluation if you trained on it. """
    # SKIP: Max length
    # if hasattr(data, "max_len") and data.max_len > 512:
    #     if logger: logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Max length ({data.max_len}) exceeds 512.")
    #     return True
    # SKIP: Continuous
    if hasattr(data, "task_type") and "continuous" in data.task_type.strip().lower():
        if logger: logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Continuous data is not supported.")
        return True
    # SKIP: Any class has < 2 samples (classification only)
    if hasattr(data, "task_type") and "classification" in data.task_type.strip().lower():
        # Use y_train/y_test if available, else fallback to .df
        y = None
        if hasattr(data, 'y_train') and hasattr(data, 'y_test'):
            y = np.concatenate([data.y_train, data.y_test])
        elif hasattr(data, 'df'):
            y = np.array(getattr(data.df, 'target', []))
        if y is not None and len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            if len(counts) == 0 or counts.min() < 2:
                if logger: logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': At least one class has <2 samples (counts: {dict(zip(unique, counts))}).")
                return True
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
