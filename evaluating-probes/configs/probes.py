# configs/probes.py
from dataclasses import dataclass, field

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass

@dataclass
class SkLearnLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the sklearn LinearProbe."""
    C: float = 3.0               # Regularization strength (default 1.0)
    penalty: str = "l2"          # 'l2' or 'none'
    solver: str = "lbfgs"        # Recommended solver for multiclass
    max_iter: int = 10000         # More than default to ensure convergence

@dataclass
class AttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the AttentionProbe."""
    lr: float = 0.01
    epochs: int = 1000
    weight_decay: float = 0.0

# --- A dictionary to easily access configs by name ---
PROBE_CONFIGS = {
    "default_linear": SkLearnLinearProbeConfig(),
    "high_reg_linear": SkLearnLinearProbeConfig(C=0.1),
    "no_reg_linear": SkLearnLinearProbeConfig(penalty="none"),
    "default_attention": AttentionProbeConfig(),
}
