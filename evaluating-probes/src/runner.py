import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from typing import Optional, Any, List

from src.data import Dataset, get_available_datasets, load_combined_classification_datasets
from src.activations import ActivationManager
from src.probes import LinearProbe, AttentionProbe
from src.logger import Logger
from src.utils import should_skip_dataset
from configs.probes import PROBE_CONFIGS

def get_probe_architecture(architecture_name: str, d_model: int):
    if architecture_name == "linear":
        return LinearProbe(d_model=d_model)
    if architecture_name == "attention":
        return AttentionProbe(d_model=d_model)
    raise ValueError(f"Unknown architecture: {architecture_name}")

def get_probe_filename_prefix(train_ds, arch_name, layer, component):
    return f"train_on_{train_ds}_{arch_name}_L{layer}_{component}"

def get_included_datasets_classification_all(logger:Logger):
    included_datasets = []
    for name in get_available_datasets():
        try:
            data = Dataset(name)
            if ("classification" in data.task_type.lower() and not should_skip_dataset(name, data, logger)):
                included_datasets.append(name)
        except Exception as e:
            if logger:
                logger.log(f"  - Skipping '{name}': {e}")
    return included_datasets

def get_combined_activations(
    datasets: List[str], layer: int, component: str, model_name: str, d_model: int, max_len: int, device: str, cache_dir: Path, logger: Logger
) -> np.ndarray:
    """Loads and concatenates cached activations for each binary dataset, regenerating any stale ones."""
    acts_list = []
    act_manager = ActivationManager(model_name, device, d_model=d_model, max_len=max_len)
    for ds in datasets:
        ds_cache_dir = cache_dir / ds
        # Always use get_activations so that any stale/corrupt cache is repaired automatically!
        ds_data = Dataset(ds)
        X_train_text, _ = ds_data.get_train_set()
        logger.log(f"  - Ensuring activations for {ds}: {ds_cache_dir}")
        arr = act_manager.get_activations(
            X_train_text, layer, component, use_cache=True, cache_dir=ds_cache_dir, logger=logger
        )
        acts_list.append(np.copy(arr))  # Load into memory
    combined = np.concatenate(acts_list, axis=0)
    return combined

def train_probe(
    model_name: str, d_model: int, train_dataset_name: str, layer: int, component: str,
    architecture_name: str, config_name: str, device: str, use_cache: bool,
    seed: int, results_dir: Path, cache_dir: Path, logger: Logger
):
    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component)
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"

    if use_cache and probe_state_path.exists():
        logger.log(f"  - Probe already trained. Skipping: {probe_state_path.name}")
        return

    logger.log("  - Probe not found in cache. Training new probe...")
    probe_save_dir.mkdir(parents=True, exist_ok=True)

    # SINGLE_ALL special case
    if train_dataset_name == "single_all":
        train_data = load_combined_classification_datasets(seed)
        X_train_text, y_train = train_data.get_train_set()
        
        included_datasets = get_included_datasets_classification_all(logger)

        train_acts = get_combined_activations(
            included_datasets, layer, component, model_name, d_model, train_data.max_len, device, cache_dir, logger
        )
    else:
        train_data = Dataset(train_dataset_name, seed=seed)
        X_train_text, y_train = train_data.get_train_set()
        act_manager = ActivationManager(model_name, device, d_model=d_model, max_len=train_data.max_len)
        train_acts_cache_dir = cache_dir / train_dataset_name
        train_acts = act_manager.get_activations(X_train_text, layer, component, use_cache, train_acts_cache_dir, logger)

    probe = get_probe_architecture(architecture_name, d_model=d_model)
    fit_params = asdict(PROBE_CONFIGS[config_name])
    probe.fit(train_acts, y_train, **fit_params)
    probe.save_state(probe_state_path)
    logger.log(f"  - ✅ Probe state saved to {probe_state_path.name}")

def evaluate_probe(
    train_dataset_name: str, eval_dataset_name: str, layer: int, component: str,
    architecture_config: dict, aggregation: str, results_dir: Path, logger: Logger,
    seed: int, model_name: str, d_model: int, device: str, use_cache: bool, cache_dir: Path
):
    architecture_name = architecture_config['name']
    config_name = architecture_config['config_name']

    logger.log("-" * 60)
    logger.log(f"🚀 Evaluating Probe:")
    logger.log(f"  - Trained on: {train_dataset_name}, Evaluated on: {eval_dataset_name}")
    logger.log(f"  - Probe: L{layer}_{component}_{architecture_name}, Aggregation: {aggregation}")

    agg_name_for_file = "attention" if architecture_name == "attention" else aggregation
    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component)
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
    eval_results_path = probe_save_dir / f"eval_on_{eval_dataset_name}__{probe_filename_base}_{agg_name_for_file}_results.json"

    if use_cache and eval_results_path.exists():
        with open(eval_results_path, 'r') as f:
            cached_data = json.load(f)
        logger.log(f"  - ✅ Loaded cached evaluation result. Metrics: {cached_data['metrics']}")
        return

    if not probe_state_path.exists():
        logger.log(f"  - ❌ ERROR: Required probe state file not found: {probe_state_path.name}. Cannot evaluate.")
        return

    probe = get_probe_architecture(architecture_name, d_model=d_model)
    probe.load_state(probe_state_path, logger)

    # SINGLE_ALL special case for eval
    if eval_dataset_name == "single_all":
        eval_data = load_combined_classification_datasets(seed)
        X_test_text, y_test = eval_data.get_test_set()

        included_datasets = get_included_datasets_classification_all(logger)
        
        test_acts = get_combined_activations(
            included_datasets, layer, component, model_name, d_model, eval_data.max_len, device, cache_dir, logger
        )
    else:
        eval_data = Dataset(eval_dataset_name, seed=seed)
        X_test_text, y_test = eval_data.get_test_set()
        act_manager = ActivationManager(model_name, device, d_model=d_model, max_len=eval_data.max_len)
        eval_acts_cache_dir = cache_dir / eval_dataset_name
        test_acts = act_manager.get_activations(X_test_text, layer, component, use_cache, eval_acts_cache_dir, logger)

    metrics = probe.score(test_acts, y_test, aggregation=agg_name_for_file)

    metadata = {
        "metrics": metrics, "train_dataset": train_dataset_name, "eval_dataset": eval_dataset_name,
        "layer": layer, "component": component, "architecture": architecture_name,
        "aggregation": agg_name_for_file, "config": asdict(PROBE_CONFIGS[config_name]), "seed": seed,
    }
    with open(eval_results_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.log(f"  - ✅ Success! New evaluation saved. Metrics: {metrics}")
