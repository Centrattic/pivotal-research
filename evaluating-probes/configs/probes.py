# configs/probes.py
from dataclasses import dataclass, field

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass

# These don't really matter now that we're using find best fit
@dataclass
class PytorchLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch LinearProbe."""
    lr: float = 5e-4
    epochs: int = 150
    batch_size: int = 512
    weight_decay: float = 0.0
    weighting_method: str = 'weighted_sampler'  # 'weighted_loss', 'weighted_sampler', or 'pcngd'
    # Add more as needed

@dataclass
class PytorchAttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch AttentionProbe."""
    lr: float = 5e-4
    epochs: int = 150
    batch_size: int = 512
    weight_decay: float = 0.0
    weighting_method: str = 'weighted_sampler'  # 'weighted_loss', 'weighted_sampler', or 'pcngd'
    # Add more as needed

@dataclass
class MassMeanProbeConfig(ProbeConfig):
    """Configuration for the Mass Mean probe (no training needed)."""
    # Note: use_iid is now determined by architecture name in runner.py
    # No parameters needed since mass-mean is computed analytically
    pass

# A dictionary to easily access configs by name. Configs are updated by -ht flag (Optuna tuning).
# The issue is we'd need separate for each dataset
PROBE_CONFIGS = {
    # Linear probe configs by aggregation
    "linear_mean": PytorchLinearProbeConfig(weighting_method='weighted_sampler'),
    "linear_max": PytorchLinearProbeConfig(weighting_method='weighted_sampler'),
    "linear_last": PytorchLinearProbeConfig(weighting_method='weighted_sampler'),
    "linear_softmax": PytorchLinearProbeConfig(),
    # High reg variants
    "linear_mean_high_reg": PytorchLinearProbeConfig(weight_decay=1e-2),
    "linear_max_high_reg": PytorchLinearProbeConfig(weight_decay=1e-2),
    "linear_last_high_reg": PytorchLinearProbeConfig(weight_decay=1e-2),
    "linear_softmax_high_reg": PytorchLinearProbeConfig(weight_decay=1e-2),
    # No reg variants
    "linear_mean_no_reg": PytorchLinearProbeConfig(weight_decay=0.0),
    "linear_max_no_reg": PytorchLinearProbeConfig(weight_decay=0.0),
    "linear_last_no_reg": PytorchLinearProbeConfig(weight_decay=0.0),
    "linear_softmax_no_reg": PytorchLinearProbeConfig(weight_decay=0.0),
    "default_linear": PytorchLinearProbeConfig(),
    "high_reg_linear": PytorchLinearProbeConfig(weight_decay=1e-2),
    "no_reg_linear": PytorchLinearProbeConfig(weight_decay=0.0),
    # Attention probe configs
    "default_attention": PytorchAttentionProbeConfig(weighting_method='weighted_sampler'),
    "high_reg_attention": PytorchAttentionProbeConfig(weight_decay=1e-2, weighting_method='weighted_sampler'),
    "no_reg_attention": PytorchAttentionProbeConfig(weight_decay=0.0, weighting_method='weighted_sampler'),
    # Mass-mean probe configs
    # Note: use_iid is determined by architecture name (mass_mean vs mass_mean_iid)
    "default_mass_mean": MassMeanProbeConfig(),
    "mass_mean_iid": MassMeanProbeConfig(),
}
