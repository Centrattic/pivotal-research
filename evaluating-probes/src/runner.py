import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from typing import Optional, Any
from src.data import Dataset
from src.activations import ActivationManager
from src.probes import LinearProbe, AttentionProbe, RidgeProbe
from src.logger import Logger
from configs.probes import PROBE_CONFIGS

# Adding as to not need to load dataset if previously over length limit
prev_train_dataset_name = None
prev_train_max_len = None

# Define a simple type hint for the logger

def get_probe_architecture(architecture_name: str, d_model: int, n_classes: Optional[int]):
    """Factory function to create a probe instance based on task type."""
    if architecture_name in ["linear", "attention"]:
        assert n_classes is not None, f"n_classes must be provided for {architecture_name}"
        if architecture_name == "linear":
            return LinearProbe(d_model=d_model, n_classes=n_classes)
        return AttentionProbe(d_model=d_model, n_classes=n_classes)
    elif architecture_name == "ridge":
        return RidgeProbe(d_model=d_model)
    raise ValueError(f"Unknown architecture: {architecture_name}")

def run_single_experiment(
    model_name: str,
    d_model: int,
    train_dataset_name: str,
    eval_dataset_name: str,
    layer: int,
    component: str,
    architecture_config: dict,
    aggregation: str,
    device: str,
    use_cache: bool,
    seed: int,
    results_dir: Path,
    cache_dir: Path,
    logger: Logger,
):
    
    global prev_train_dataset_name, prev_train_max_len

    if (prev_train_dataset_name == train_dataset_name):
        # logger.log(f"  - ⏭️  Skipping training dataset '{train_dataset_name}' (max_len: {prev_train_max_len}), exceeds 512 token limit.")
        return

    logger.log("-" * 60)
    logger.log(f"  - Evaluating on: {eval_dataset_name}, Layer: {layer}, Component: {component}")
    logger.log(f"  - Architecture: {architecture_config['name']}, Config: {architecture_config['config_name']}, Aggregation: {aggregation}")

    architecture_name = architecture_config['name']
    config_name = architecture_config['config_name']

    # Data loading and inspection
    train_data = Dataset(train_dataset_name, seed=seed)
    task_type = train_data.task_type
    n_classes = train_data.n_classes
    max_len = train_data.max_len

    if max_len > 512:
        prev_train_dataset_name = train_dataset_name
        prev_train_max_len = max_len
        logger.log(f"  - ⏭️  Skipping training dataset '{train_dataset_name}' (max_len: {train_data.max_len}), exceeds 512 token limit.")
        return
    
    logger.log(f"  - Detected task: {task_type} (n_classes={n_classes}, max_len={max_len})")

    # Conditional logic for task type
    if "Classification" not in task_type and architecture_name != "ridge":
        logger.log(f"  - ⏭️  Skipping classification architecture '{architecture_name}' for non-classification task.")
        return
    if task_type == "Continuous data" and architecture_name == "ridge":
        if aggregation not in ["mean", "last_token"]:
            logger.log(f"  - ⏭️  Skipping aggregation '{aggregation}' not suitable for RidgeProbe.")
            return
    elif task_type == "Continuous data":
        logger.log(f"  - ⏭️  Skipping non-regression architecture for continuous task.")
        return
        
    if architecture_name == "attention" and aggregation != "mean":
        logger.log(f"  - ⏭️  Skipping redundant aggregation '{aggregation}' for attention architecture.")
        return

    # Setup and probe caching
    agg_name = "attention" if architecture_name == "attention" else aggregation
    probe_filename_base = f"train_on_{train_dataset_name}_{architecture_name}_L{layer}_{component}_{agg_name}"
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"

    probe = get_probe_architecture(architecture_name, d_model=d_model, n_classes=n_classes)
    act_manager = ActivationManager(model_name, device, d_model=d_model, max_len=max_len)

    if use_cache and probe_state_path.exists():
        probe.load_state(probe_state_path, logger) # Pass logger to load_state
    else:
        logger.log("  - Probe not found in cache. Training new probe...")
        probe_save_dir.mkdir(parents=True, exist_ok=True)
        X_train_text, y_train = train_data.get_train_set()
        
        train_acts_cache_dir = cache_dir / train_dataset_name
        train_acts = act_manager.get_activations(X_train_text, layer, component, use_cache, train_acts_cache_dir, logger)
        
        fit_params = asdict(PROBE_CONFIGS[config_name])
        probe.fit(train_acts, y_train, aggregation=aggregation, **fit_params)
        probe.save_state(probe_state_path)

    # Evaluation
    eval_data = Dataset(eval_dataset_name, seed=seed)
    if eval_data.max_len > 512:
        logger.log(f"  - ⏭️  Skipping evaluation dataset '{eval_dataset_name}' (max_len: {eval_data.max_len}), exceeds 512 token limit.")
        return
        
    logger.log(f"  - Evaluating on test set of '{eval_dataset_name}'...")
    X_test_text, y_test = eval_data.get_test_set()

    eval_acts_cache_dir = cache_dir / eval_dataset_name
    test_acts = act_manager.get_activations(X_test_text, layer, component, use_cache, eval_acts_cache_dir, logger)
    
    metrics = probe.score(test_acts, y_test, aggregation=aggregation)
    
    # Save evaluation results
    eval_results_filename = f"eval_on_{eval_dataset_name}__{probe_filename_base}_results.json"
    
    metadata = {
        "metrics": metrics,
        "train_dataset": train_dataset_name,
        "eval_dataset": eval_dataset_name,
        "layer": layer,
        "component": component,
        "architecture": architecture_name,
        "aggregation": agg_name,
        "config": asdict(PROBE_CONFIGS[config_name]),
        "seed": seed,
    }
    
    with open(probe_save_dir / eval_results_filename, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.log(f"  - ✅ Success! Metrics: {metrics}")