
import numpy as np
import json
from typing import List, Any
from src.logger import Logger
from pathlib import Path
from dataclasses import asdict
from src.probes import LinearProbe, AttentionProbe, MassMeanProbe
from configs.probes import PROBE_CONFIGS

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
    if hasattr(config, 'aggregation'):
        return config.aggregation
    elif architecture_name == "attention":
        return "attention"
    elif architecture_name in ["mass_mean", "mass_mean_iid"]:
        return "mean"
    else:
        return "mean"  # Default fallback


def get_probe_architecture(architecture_name: str, d_model: int, device, config: dict):
    """Create probe architecture with all config parameters passed directly."""
    if architecture_name == "linear":
        return LinearProbe(d_model=d_model, device=device, **config)
    if architecture_name == "attention":
        return AttentionProbe(d_model=d_model, device=device, **config)
    if architecture_name in ["mass_mean", "mass_mean_iid"]:
        # Mass-mean probes need use_iid parameter from config
        return MassMeanProbe(d_model=d_model, device=device, **config)
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
