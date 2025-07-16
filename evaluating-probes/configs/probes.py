# configs/probes.py
from dataclasses import dataclass, field

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass

@dataclass
class SkLearnLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the sklearn LinearProbe."""
    C: float = 1              # Regularization strength (default 1.0), higher reg if more params than data samples (magic of reg?)
    penalty: str = "l2"          # 'l2' or 'none'
    solver: str = "lbfgs"        # Recommended solver for multiclass
    max_iter: int = 2000         # More than default to ensure convergence
    class_weight="balanced"

@dataclass
class AttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the AttentionProbe."""
    lr: float = 0.01
    epochs: int = 1000
    weight_decay: float = 0.0

@dataclass
class TwoStageConfig(ProbeConfig):
    stage1_kwargs: dict = field(default_factory=dict)
    stage2_kwargs: dict = field(default_factory=dict)



# --- A dictionary to easily access configs by name ---
PROBE_CONFIGS = {
    "default_linear": SkLearnLinearProbeConfig(),
    "high_reg_linear": SkLearnLinearProbeConfig(C=0.1),
    "no_reg_linear": SkLearnLinearProbeConfig(penalty="none"),
    "default_attention": AttentionProbeConfig(),
    "two_stage_defaults": TwoStageConfig(),

}
