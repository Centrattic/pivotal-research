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
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 512
    weight_decay: float = 0.0
    weighting_method: str = 'weighted_loss'  # 'weighted_loss', 'weighted_sampler', or 'pcngd'
    # Add more as needed

@dataclass
class PytorchAttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch AttentionProbe."""
    lr: float = 1e-3
    epochs: int = 75
    batch_size: int = 512
    weight_decay: float = 0.0
    weighting_method: str = 'weighted_loss'  # 'weighted_loss', 'weighted_sampler', or 'pcngd'
    # Add more as needed

# A dictionary to easily access configs by name. Configs are updated by -ht flag (Optuna tuning).
# The issue is we'd need separate for each dataset
PROBE_CONFIGS = {
    # Linear probe configs by aggregation
    "linear_mean": PytorchLinearProbeConfig(lr=0.0007395535622979691, weight_decay=4.649132978175384e-06, weighting_method='weighted_loss'),
    "linear_max": PytorchLinearProbeConfig(weighting_method='weighted_loss'),
    "linear_last": PytorchLinearProbeConfig(lr= 0.00039989367421521075, weight_decay=7.771688681156908e-08, weighting_method='weighted_loss'),
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
    "default_attention": PytorchAttentionProbeConfig(lr=0.0003886838187159334, weight_decay=1.2660720185538272e-06, weighting_method='weighted_loss'),
    "high_reg_attention": PytorchAttentionProbeConfig(weight_decay=1e-2, weighting_method='weighted_loss'),
    "no_reg_attention": PytorchAttentionProbeConfig(weight_decay=0.0, weighting_method='weighted_loss'),
}
