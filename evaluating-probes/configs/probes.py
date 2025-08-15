# configs/probes.py
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Set sklearn to use only 1 thread for CPU-based training
# os.environ["OMP_NUM_THREADS"] = "1"

@dataclass
class ProbeJob:
    """Represents a single probe training/evaluation job."""
    # Basic identifiers
    experiment_name: str
    train_dataset: str
    eval_datasets: List[str]
    layer: int
    component: str
    seed: int

    # Probe configuration
    architecture_name: str  # sklearn_linear, linear, sae, attention, act_sim, mass_mean
    probe_config: Any  # Union[SklearnLinearProbeConfig, PytorchAttentionProbeConfig, SAEProbeConfig, MassMeanProbeConfig, ActivationSimilarityProbeConfig]

    # Data configuration
    on_policy: bool  # Whether this is an on-policy probe - must be specified
    rebuild_config: Optional[Dict] = None

    # Training configuration
    train_size: float = 0.75
    val_size: float = 0.10
    test_size: float = 0.15

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass


@dataclass
class SklearnLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the sklearn LinearProbe."""
    aggregation: str = None  # mean, max, last, softmax
    solver: str = "liblinear"  # liblinear, lbfgs, newton-cg, sag, saga
    C: float = 1.0  # Inverse of regularization strength
    max_iter: int = 1000
    class_weight: str = "balanced"  # balanced, None

@dataclass
class PytorchAttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch AttentionProbe."""
    lr: float = 5e-4
    epochs: int = 100
    batch_size: int = 1024  # H100: 2560, H200: 1024, A6000: 32
    weight_decay: float = 0.0
    verbose: bool = True
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class SAEProbeConfig(ProbeConfig):
    """Hyperparameters for the SAE Probe."""
    aggregation: str = None  # mean, max, last, softmax
    model_name: str = None,
    layer: int = 20
    sae_id: str = None  # Specific SAE ID to use
    top_k_features: int = 3584 # set to same as residual stream!
    lr: float = 5e-4
    epochs: int = 100
    encoding_batch_size: int = 1280  # Batch size for SAE encoding (memory intensive) - H100: 100
    training_batch_size: int = 2560  # Batch size for probe training - H100: 512
    weight_decay: float = 0.0
    verbose: bool = True
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4


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
    # Default sklearn configs (for backward compatibility)
    "default_sklearn_linear": SklearnLinearProbeConfig(),
    "high_reg_sklearn_linear": SklearnLinearProbeConfig(C=0.01),
    "no_reg_sklearn_linear": SklearnLinearProbeConfig(C=100.0),

    # Attention probe configs
    "attention": PytorchAttentionProbeConfig(),

    # SAE probe configs - specific SAE IDs with different batch sizes
    # using gemma-scope-9b-pt-res as done in Kantamneni, not gemma-2-9b-it
    "sae_16k_l0_408_last": SAEProbeConfig(
        aggregation="last",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_16k/average_l0_408",
    ),
    "sae_16k_l0_408_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_16k/average_l0_408",
    ),
    "sae_16k_l0_408_max": SAEProbeConfig(
        aggregation="max",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_16k/average_l0_408",
    ),
    "sae_16k_l0_408_softmax": SAEProbeConfig(
        aggregation="softmax",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_16k/average_l0_408",
    ),

    "sae_262k_l0_259_last": SAEProbeConfig(
        aggregation="last",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_262k/average_l0_259",
    ),
    "sae_262k_l0_259_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_262k/average_l0_259",
    ),
    "sae_262k_l0_259_max": SAEProbeConfig(
        aggregation="max",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_262k/average_l0_259",
    ),
    "sae_262k_l0_259_softmax": SAEProbeConfig(
        aggregation="softmax",
        model_name="google/gemma-2-9b",
        sae_id="layer_20/width_262k/average_l0_259",
    ),

    "sae_llama33b_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        top_k_features=128,
        encoding_batch_size=1280,
        training_batch_size=2560,
    ),
    
    # Qwen3 family (SAE Lens transcoders)
    # Qwen3 0.6B (layer_15)
    "sae_qwen3_0.6b_last": SAEProbeConfig(
        aggregation="last",
        model_name="Qwen/Qwen3-0.6B",
        sae_id="layer_15",
    ),
    "sae_qwen3_0.6b_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="Qwen/Qwen3-0.6B",
        sae_id="layer_15",
    ),
    "sae_qwen3_0.6b_max": SAEProbeConfig(
        aggregation="max",
        model_name="Qwen/Qwen3-0.6B",
        sae_id="layer_15",
    ),
    "sae_qwen3_0.6b_softmax": SAEProbeConfig(
        aggregation="softmax",
        model_name="Qwen/Qwen3-0.6B",
        sae_id="layer_15",
    ),

    # Qwen3 1.7B (layer_15)
    "sae_qwen3_1.7b_last": SAEProbeConfig(
        aggregation="last",
        model_name="Qwen/Qwen3-1.7B",
        sae_id="layer_15",
    ),
    "sae_qwen3_1.7b_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="Qwen/Qwen3-1.7B",
        sae_id="layer_15",
    ),
    "sae_qwen3_1.7b_max": SAEProbeConfig(
        aggregation="max",
        model_name="Qwen/Qwen3-1.7B",
        sae_id="layer_15",
    ),
    "sae_qwen3_1.7b_softmax": SAEProbeConfig(
        aggregation="softmax",
        model_name="Qwen/Qwen3-1.7B",
        sae_id="layer_15",
    ),

    # Qwen3 4B (layer_18)
    "sae_qwen3_4b_last": SAEProbeConfig(
        aggregation="last",
        model_name="Qwen/Qwen3-4B",
        sae_id="layer_18",
    ),
    "sae_qwen3_4b_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="Qwen/Qwen3-4B",
        sae_id="layer_18",
    ),
    "sae_qwen3_4b_max": SAEProbeConfig(
        aggregation="max",
        model_name="Qwen/Qwen3-4B",
        sae_id="layer_18",
    ),
    "sae_qwen3_4b_softmax": SAEProbeConfig(
        aggregation="softmax",
        model_name="Qwen/Qwen3-4B",
        sae_id="layer_18",
    ),

    # Qwen3 8B (layer_18)
    "sae_qwen3_8b_last": SAEProbeConfig(
        aggregation="last",
        model_name="Qwen/Qwen3-8B",
        sae_id="layer_18",
    ),
    "sae_qwen3_8b_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="Qwen/Qwen3-8B",
        sae_id="layer_18",
    ),
    "sae_qwen3_8b_max": SAEProbeConfig(
        aggregation="max",
        model_name="Qwen/Qwen3-8B",
        sae_id="layer_18",
    ),
    "sae_qwen3_8b_softmax": SAEProbeConfig(
        aggregation="softmax",
        model_name="Qwen/Qwen3-8B",
        sae_id="layer_18",
    ),

    # Qwen3 14B (layer_24)
    "sae_qwen3_14b_last": SAEProbeConfig(
        aggregation="last",
        model_name="Qwen/Qwen3-14B",
        sae_id="layer_24",
    ),
    "sae_qwen3_14b_mean": SAEProbeConfig(
        aggregation="mean",
        model_name="Qwen/Qwen3-14B",
        sae_id="layer_24",
    ),
    "sae_qwen3_14b_max": SAEProbeConfig(
        aggregation="max",
        model_name="Qwen/Qwen3-14B",
        sae_id="layer_24",
    ),
    "sae_qwen3_14b_softmax": SAEProbeConfig(
        aggregation="softmax",
        model_name="Qwen/Qwen3-14B",
        sae_id="layer_24",
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
