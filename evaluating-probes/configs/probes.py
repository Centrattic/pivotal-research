# configs/probes.py
from dataclasses import dataclass, field

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass

@dataclass
class PytorchLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch LinearProbe."""
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    weight_decay: float = 0.0
    # Add more as needed

@dataclass
class PytorchAttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch AttentionProbe."""
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    weight_decay: float = 0.0
    # Add more as needed

@dataclass
class TwoStageConfig(ProbeConfig):
    stage1_kwargs: dict = field(default_factory=dict)
    stage2_kwargs: dict = field(default_factory=dict)

# --- A dictionary to easily access configs by name ---
PROBE_CONFIGS = {
    "default_linear": PytorchLinearProbeConfig(),
    "high_reg_linear": PytorchLinearProbeConfig(weight_decay=1e-2),
    "no_reg_linear": PytorchLinearProbeConfig(weight_decay=0.0),
    "default_attention": PytorchAttentionProbeConfig(),
    "high_reg_attention": PytorchAttentionProbeConfig(weight_decay=1e-2),
    "no_reg_attention": PytorchAttentionProbeConfig(weight_decay=0.0),
    "two_stage_defaults": TwoStageConfig(),
}
