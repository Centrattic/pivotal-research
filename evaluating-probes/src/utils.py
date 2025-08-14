import numpy as np
import json
from typing import List, Any
from src.logger import Logger
from pathlib import Path
from dataclasses import asdict
from configs.probes import PROBE_CONFIGS
from src.data import Dataset
import pandas as pd
import os


def dump_loss_history(
    losses: List[float],
    out_path: Path,
    logger: Logger | None = None,
):
    """
     Write `[loss0, loss1, ...]` as pretty-printed JSON.
     """
    with open(out_path, "w") as fp:
        json.dump(losses, fp, indent=2)
    if logger:
        logger.log(f"  - 😋 Loss history saved in {out_path.name}")


def extract_aggregation_from_config(
    config_name: str,
    architecture_name: str,
) -> str:
    """Extract aggregation from config for backward compatibility."""
    config = PROBE_CONFIGS[config_name]
    if hasattr(config, 'aggregation'):
        return config.aggregation
    elif architecture_name == "attention":
        return "attention"
    elif architecture_name.startswith("mass_mean"):
        return "mass_mean"
    else:
        return "mean"  # Default fallback


def get_probe_architecture(
    architecture_name: str,
    d_model: int,
    device,
    config: dict,
):
    """Create probe architecture with filtered config parameters."""
    # Import probes inside function to avoid circular imports
    from src.probes import (
        LinearProbe,
        SklearnLinearProbe,
        AttentionProbe,
        MassMeanProbe,
        ActivationSimilarityProbe,
        SAEProbe
    )

    if architecture_name == "sklearn_linear":
        return SklearnLinearProbe(
            d_model=d_model,
            device=device,
            **config,
        )
    elif architecture_name == "linear":
        return LinearProbe(
            d_model=d_model,
            device=device,
            **config,
        )
    elif architecture_name == "attention":
        return AttentionProbe(
            d_model=d_model,
            device=device,
            **config,
        )
    elif architecture_name.startswith("sae"):
        return SAEProbe(
            d_model=d_model,
            device=device,
            **config,
        )
    elif architecture_name.startswith("mass_mean"):
        return MassMeanProbe(
            d_model=d_model,
            device=device,
            **config,
        )
    elif architecture_name.startswith("act_sim"):
        # Activation similarity probes - filter out batch_size from constructor args
        filtered_config = {k: v for k, v in config.items() if k != 'batch_size'}
        return ActivationSimilarityProbe(
            d_model=d_model,
            device=device,
            **filtered_config,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")


def get_probe_filename_prefix(
    train_ds,
    arch_name,
    layer,
    component,
    config_name,
):
    # Use config_name instead of architecture name for better organization
    agg_name = extract_aggregation_from_config(config_name, arch_name)

    base_prefix = f"train_on_{train_ds}_{config_name}_L{layer}_{component}"

    return base_prefix


def rebuild_suffix(
    rebuild_config,
):
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


def resample_params_to_str(
    params,
):
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


def get_dataset(
    name,
    model,
    device,
    seed,
):
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


def generate_llm_upsampling_configs(
    n_real_neg,
    n_real_pos_list,
    upsampling_factors,
    seed,
):
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
            configs.append(
                {
                    'llm_upsampling': True,
                    'n_real_neg': n_real_neg,
                    'n_real_pos': n_real_pos,
                    'upsampling_factor': factor,
                    'seed': seed  # Override global seed for LLM experiments
                }
            )
    return configs
