# configs/probes.py
from dataclasses import dataclass, field

@dataclass
class ProbeConfig:
    """Base class for probe configurations."""
    pass

@dataclass
class PytorchLinearProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch LinearProbe."""
    aggregation: str = "mean"  # mean, max, last, softmax
    lr: float = 5e-4
    epochs: int = 100
    batch_size: int = 1024 # H100: 2048, H200: 800, A6000: 32
    weight_decay: float = 0.0
    weighting_method: str = 'weighted_sampler'  # 'weighted_loss', 'weighted_sampler', or 'pcngd'
    # Add more as needed

@dataclass
class PytorchAttentionProbeConfig(ProbeConfig):
    """Hyperparameters for the PyTorch AttentionProbe."""
    lr: float = 5e-4
    epochs: int = 100
    batch_size: int = 1024 # H100: 2560, H200: 1024, A6000: 32
    weight_decay: float = 0.0
    weighting_method: str = 'weighted_sampler'  # 'weighted_loss', 'weighted_sampler', or 'pcngd'
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
    weighting_method: str = 'weighted_sampler'  # 'weighted_loss', 'weighted_sampler', or 'pcngd'

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
    # Linear probe configs - now with aggregation as a parameter
    "linear_mean": PytorchLinearProbeConfig(aggregation="mean", weighting_method='weighted_sampler'),
    "linear_max": PytorchLinearProbeConfig(aggregation="max", weighting_method='weighted_sampler'),
    "linear_last": PytorchLinearProbeConfig(aggregation="last", weighting_method='weighted_sampler'),
    "linear_softmax": PytorchLinearProbeConfig(aggregation="softmax", weighting_method='weighted_sampler'),
    # High reg variants
    "linear_mean_high_reg": PytorchLinearProbeConfig(aggregation="mean", weight_decay=1e-2),
    "linear_max_high_reg": PytorchLinearProbeConfig(aggregation="max", weight_decay=1e-2),
    "linear_last_high_reg": PytorchLinearProbeConfig(aggregation="last", weight_decay=1e-2),
    "linear_softmax_high_reg": PytorchLinearProbeConfig(aggregation="softmax", weight_decay=1e-2),
    # No reg variants
    "linear_mean_no_reg": PytorchLinearProbeConfig(aggregation="mean", weight_decay=0.0),
    "linear_max_no_reg": PytorchLinearProbeConfig(aggregation="max", weight_decay=0.0),
    "linear_last_no_reg": PytorchLinearProbeConfig(aggregation="last", weight_decay=0.0),
    "linear_softmax_no_reg": PytorchLinearProbeConfig(aggregation="softmax", weight_decay=0.0),
    # Default configs (for backward compatibility)
    "default_linear": PytorchLinearProbeConfig(),
    "high_reg_linear": PytorchLinearProbeConfig(weight_decay=1e-2),
    "no_reg_linear": PytorchLinearProbeConfig(weight_decay=0.0),
    # Attention probe configs
    "attention": PytorchAttentionProbeConfig(weighting_method='weighted_sampler'),
    "default_attention": PytorchAttentionProbeConfig(weighting_method='weighted_sampler'),
    "high_reg_attention": PytorchAttentionProbeConfig(weight_decay=1e-2, weighting_method='weighted_sampler'),
    "no_reg_attention": PytorchAttentionProbeConfig(weight_decay=0.0, weighting_method='weighted_sampler'),
    # SAE probe configs - specific SAE IDs with different batch sizes
    # using gemma-scope-9b-pt-res as done in Kantamneni, not gemma-2-9b-it
    "sae_16k_l0_408_last": SAEProbeConfig(
        aggregation="last", # no matter right now
        sae_id="layer_20/width_16k/average_l0_408", # for some reason lo_189 doesn't exist! 
        weighting_method='weighted_sampler'
    ),
    "sae_262k_l0_259_last": SAEProbeConfig(
        aggregation="last", 
        sae_id="layer_20/width_262k/average_l0_259",
        # encoding_batch_size=1024,  # H100: 1024
        # training_batch_size=256,   # H100: 256
        weighting_method='weighted_sampler'
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
