# configs/probes.py
import os
from dataclasses import dataclass, field

# Set sklearn to use only 1 thread for CPU-based training
# os.environ["OMP_NUM_THREADS"] = "1"

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass

@dataclass
class SklearnLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the sklearn LinearProbe."""
    aggregation: str = "mean"  # mean, max, last, softmax
    solver: str = "liblinear"  # liblinear, lbfgs, newton-cg, sag, saga
    C: float = 1.0  # Inverse of regularization strength
    max_iter: int = 1000
    class_weight: str = "balanced"  # balanced, None
    random_state: int = 42
    # Add more as needed

@dataclass
class PytorchLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch LinearProbe."""
    aggregation: str = "mean"  # mean, max, last, softmax
    lr: float = 5e-4
    epochs: int = 100
    batch_size: int = 1024 # H100: 2048, H200: 800, A6000: 32
    weight_decay: float = 0.0
    # Add more as needed

@dataclass
class PytorchAttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch AttentionProbe."""
    lr: float = 5e-4
    epochs: int = 100
    batch_size: int = 1024 # H100: 2560, H200: 1024, A6000: 32
    weight_decay: float = 0.0
    # Add more as needed

@dataclass
class SAEProbeConfig(ProbeConfig):
    """Hyperparameters for the SAE Probe."""
    aggregation: str = "mean"  # mean, max, last, softmax
    model_name: str = "gemma-2-9b"
    layer: int = 20
    sae_id: str = None  # Specific SAE ID to use
    top_k_features: int = 128
    lr: float = 5e-4
    epochs: int = 100
    encoding_batch_size: int = 1280  # Batch size for SAE encoding (memory intensive) - H100: 100
    training_batch_size: int = 2560   # Batch size for probe training - H100: 512
    weight_decay: float = 0.0

@dataclass
class MassMeanProbeConfig(ProbeConfig):
    """Configuration for the Mass Mean probe (no training needed)."""
    use_iid: bool = False  # Whether to use IID version (Fisher's LDA)
    batch_size: int = 1280  # Batch size for processing - H100: 1024
    # No other parameters needed since mass-mean is computed analytically

@dataclass
class ActivationSimilarityProbeConfig(ProbeConfig):
    """Configuration for the Activation Similarity probe (no training needed)."""
    aggregation: str = "mean"  # mean, max, last, softmax
    batch_size: int = 1280  # Batch size for processing - H100: 1024
    # No other parameters needed since activation similarity is computed analytically

# A dictionary to easily access configs by name. Configs are updated by -ht flag (Optuna tuning).
# The issue is we'd need separate for each dataset
PROBE_CONFIGS = {
    # Sklearn linear probe configs - new primary linear probe
    "sklearn_linear_mean": SklearnLinearProbeConfig(aggregation="mean"),
    "sklearn_linear_max": SklearnLinearProbeConfig(aggregation="max"),
    "sklearn_linear_last": SklearnLinearProbeConfig(aggregation="last"),
    "sklearn_linear_softmax": SklearnLinearProbeConfig(aggregation="softmax"),
    # High reg variants
    "sklearn_linear_mean_high_reg": SklearnLinearProbeConfig(aggregation="mean", C=0.01),
    "sklearn_linear_max_high_reg": SklearnLinearProbeConfig(aggregation="max", C=0.01),
    "sklearn_linear_last_high_reg": SklearnLinearProbeConfig(aggregation="last", C=0.01),
    "sklearn_linear_softmax_high_reg": SklearnLinearProbeConfig(aggregation="softmax", C=0.01),
    # No reg variants
    "sklearn_linear_mean_no_reg": SklearnLinearProbeConfig(aggregation="mean", C=100.0),
    "sklearn_linear_max_no_reg": SklearnLinearProbeConfig(aggregation="max", C=100.0),
    "sklearn_linear_last_no_reg": SklearnLinearProbeConfig(aggregation="last", C=100.0),
    "sklearn_linear_softmax_no_reg": SklearnLinearProbeConfig(aggregation="softmax", C=100.0),
    # Default sklearn configs (for backward compatibility)
    "default_sklearn_linear": SklearnLinearProbeConfig(),
    "high_reg_sklearn_linear": SklearnLinearProbeConfig(C=0.01),
    "no_reg_sklearn_linear": SklearnLinearProbeConfig(C=100.0),
    
    # Linear probe configs - now with aggregation as a parameter (legacy PyTorch)
    "linear_mean": PytorchLinearProbeConfig(aggregation="mean"),
    "linear_max": PytorchLinearProbeConfig(aggregation="max"),
    "linear_last": PytorchLinearProbeConfig(aggregation="last"),
    "linear_softmax": PytorchLinearProbeConfig(aggregation="softmax"),
    
    # Attention probe configs
    "attention": PytorchAttentionProbeConfig(),

    # SAE probe configs - specific SAE IDs with different batch sizes
    # using gemma-scope-9b-pt-res as done in Kantamneni, not gemma-2-9b-it
    "sae_16k_l0_408_last": SAEProbeConfig(
        aggregation="last", # no matter right now
        sae_id="layer_20/width_16k/average_l0_408", 
    ),
    "sae_262k_l0_259_last": SAEProbeConfig(
        aggregation="last", 
        sae_id="layer_20/width_262k/average_l0_259",
        # encoding_batch_size=1024,  # H100: 1024
        # training_batch_size=256,   # H100: 256
    ),
    # Mass-mean probe configs (IID functionality disabled due to numerical instability)
    "mass_mean": MassMeanProbeConfig(use_iid=False),
    # "mass_mean_iid": MassMeanProbeConfig(use_iid=True),  # IID functionality disabled
    "default_mass_mean": MassMeanProbeConfig(use_iid=False),
    # Activation similarity probe configs
    "act_sim_mean": ActivationSimilarityProbeConfig(aggregation="mean"),
    "act_sim_max": ActivationSimilarityProbeConfig(aggregation="max"),
    "act_sim_last": ActivationSimilarityProbeConfig(aggregation="last"),
    "act_sim_softmax": ActivationSimilarityProbeConfig(aggregation="softmax"),
    "default_act_sim": ActivationSimilarityProbeConfig(aggregation="mean"),
}
