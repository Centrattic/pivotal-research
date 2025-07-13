import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from typing import Optional, Any
from src.data import Dataset
from src.activations import ActivationManager
from src.probes import LinearProbe, AttentionProbe
from src.logger import Logger
from configs.probes import PROBE_CONFIGS

# Adding as to not need to load dataset if previously skipped, may not need separate train and eval skips
prev_train_skip_dataset_name = None
prev_eval_skip_dataset_name = None

# Define a simple type hint for the logger

def get_probe_architecture(architecture_name: str, d_model: int):
    """Factory function to create a probe instance."""
    if architecture_name == "linear":
        return LinearProbe(d_model=d_model)
    if architecture_name == "attention":
        return AttentionProbe(d_model=d_model)
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
    
    architecture_name = architecture_config['name']
    config_name = architecture_config['config_name']
    
    # Skip experiment if previously skipped.
    global prev_train_skip_dataset_name, prev_eval_skip_dataset_name

    if (prev_train_skip_dataset_name == train_dataset_name or prev_eval_skip_dataset_name == eval_dataset_name):
        return
    
    # Skip experiment if previously completed.
    agg_name = "attention" if architecture_name == "attention" else aggregation
    probe_filename_base = f"train_on_{train_dataset_name}_{architecture_name}_L{layer}_{component}_{agg_name}"
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"

    eval_results_path = probe_save_dir / f"eval_on_{eval_dataset_name}__{probe_filename_base}_results.json"

    # --- Check for Cached Evaluation Result ---
    if use_cache and eval_results_path.exists():
        with open(eval_results_path, 'r') as f:
            cached_data = json.load(f)
        logger.log(f"  - ✅ Loaded cached evaluation result. Metrics: {cached_data['metrics']}")
        return

    logger.log("-" * 60)
    logger.log(f"  - Evaluating on: {eval_dataset_name}, Layer: {layer}, Component: {component}")
    logger.log(f"  - Architecture: {architecture_name}, Config: {config_name}, Aggregation: {aggregation}")

    # Data loading and inspection
    train_data = Dataset(train_dataset_name, seed=seed)
    task_type = train_data.task_type
    n_classes = train_data.n_classes

    eval_data = Dataset(eval_dataset_name, seed=seed)

    # ToDo: clean up these skips in the future! For all skips should add prevs to make output text in output.log much more concise.

    # 1. Skip if training on a continuous dataset
    if "Continuous" in train_data.task_type:
        prev_train_skip_dataset_name = train_dataset_name
        logger.log(f"  - ⏭️  Skipping job: Training on continuous data ('{train_dataset_name}') is not supported.")
        return

    # 2. Skip if task types or class counts are mismatched
    if train_data.task_type != eval_data.task_type:
        prev_eval_skip_dataset_name = eval_dataset_name
        logger.log(f"  - ⏭️  Skipping job: Mismatched task types (Train: {train_data.task_type}, Eval: {eval_data.task_type}).")
        return
    
    if train_data.n_classes != eval_data.n_classes:
        prev_eval_skip_dataset_name = eval_dataset_name
        logger.log(f"  - ⏭️  Skipping job: Mismatched number of classes (Train: {train_data.n_classes}, Eval: {eval_data.n_classes}).")
        return

    # 3. Skip if dataset length is too long
    if train_data.max_len > 512:
        prev_train_skip_dataset_name = train_dataset_name
        logger.log(f"  - ⏭️  Skipping training dataset '{train_dataset_name}' (max_len: {train_data.max_len}), exceeds 512 token limit.")
        return

    if eval_data.max_len > 512: # The global skip doesn't help too much with this because of experiment ordering.
        prev_eval_skip_dataset_name = eval_dataset_name
        logger.log(f"  - ⏭️  Skipping evaluation dataset '{eval_dataset_name}' (max_len: {eval_data.max_len}), exceeds 512 token limit.")
        return

    logger.log(f"  - Detected task: {task_type} (n_classes={n_classes}, max_len={train_data.max_len})")

    # Conditional logic for task type        
    if architecture_name == "attention" and aggregation != "mean":
        logger.log(f"  - ⏭️  Skipping redundant aggregation '{aggregation}' for attention architecture.")
        return

    # Probe caching
    probe = get_probe_architecture(architecture_name, d_model=d_model)
    max_len = max(train_data.max_len, eval_data.max_len)
    act_manager = ActivationManager(model_name, device, d_model=d_model, max_len=max_len)

    if use_cache and probe_state_path.exists():
        probe.load_state(probe_state_path, logger) # Pass logger to load_state
    else:
        logger.log("  - Probe not found in cache. Training new probe...")
        probe_save_dir.mkdir(parents=True, exist_ok=True)
        X_train_text, y_train = train_data.get_train_set()
        
        # Create an ActivationManager specifically for the training dataset's max_len
        train_act_manager = ActivationManager(model_name, device, d_model=d_model, max_len=train_data.max_len)
        train_acts_cache_dir = cache_dir / train_dataset_name
        train_acts = train_act_manager.get_activations(X_train_text, layer, component, use_cache, train_acts_cache_dir, logger)
        
        fit_params = asdict(PROBE_CONFIGS[config_name])
        probe.fit(train_acts, y_train, aggregation=aggregation, **fit_params)
        probe.save_state(probe_state_path)

    # Evaluation        
    logger.log(f"  - Evaluating on test set of '{eval_dataset_name}'...")
    X_test_text, y_test = eval_data.get_test_set()

    # Create an ActivationManager specifically for the evaluation dataset's max_len
    eval_act_manager = ActivationManager(model_name, device, d_model=d_model, max_len=eval_data.max_len)
    eval_acts_cache_dir = cache_dir / eval_dataset_name
    test_acts = eval_act_manager.get_activations(X_test_text, layer, component, use_cache, eval_acts_cache_dir, logger)
    
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