# configs/probes.py
from dataclasses import dataclass, field

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass

@dataclass
class LinearProbeConfig(ProbeConfig):
    """Hyperparameters for the LinearProbe."""
    lr: float = 0.01
    epochs: int = 100
    weight_decay: float = 0.01

@dataclass
class AttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the AttentionProbe."""
    lr: float = 0.01
    epochs: int = 100
    weight_decay: float = 0.0

# --- A dictionary to easily access configs by name ---
PROBE_CONFIGS = {
    "default_linear": LinearProbeConfig(),
    "high_lr_linear": LinearProbeConfig(lr=0.1),
    "long_train_linear": LinearProbeConfig(epochs=250),
    "default_attention": AttentionProbeConfig(),
}
