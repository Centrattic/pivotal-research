import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from typing import Optional, Any, List

from src.data import Dataset, get_available_datasets, load_combined_classification_datasets
# from src.activations import ActivationManager
from src.probes import LinearProbe, AttentionProbe
from src.logger import Logger
from configs.probes import PROBE_CONFIGS
from dataclasses import asdict
import torch

def get_probe_architecture(architecture_name: str, d_model: int, device, aggregation: str = "mean"):
    if architecture_name == "linear":
        return LinearProbe(d_model=d_model, device=device, aggregation=aggregation)
    if architecture_name == "attention":
        return AttentionProbe(d_model=d_model, device=device)
    raise ValueError(f"Unknown architecture: {architecture_name}")

def get_probe_filename_prefix(train_ds, arch_name, aggregation, layer, component):
    # For attention probes, aggregation is not used in the model, but we keep it in filename for consistency
    return f"train_on_{train_ds}_{arch_name}_{aggregation}_L{layer}_{component}"

# def get_included_datasets_classification_all(logger:Logger):
#     included_datasets = []
#     for name in get_available_datasets():
#         try:
#             data = Dataset(name) # only binary classification allowed!
#             if ("binary" in data.task_type.lower() and not should_skip_dataset(name, data, logger)):
#                 included_datasets.append(name)
#         except Exception as e:
#             if logger:
#                 logger.log(f"  - Skipping '{name}': {e}")
#     return included_datasets

# def get_combined_activations(
#     datasets: List[str], layer: int, component: str,
#     model: Any, d_model: int, final_max_len: int, device: str,
#     cache_dir: Path, logger: Logger,
#     seed: int,  # <-- ADDED
#     split: str  # <-- ADDED ('train' or 'test')
# ) -> np.ndarray:
#     """..."""
#     acts_list = []
#     for ds in datasets:
#         # Pass the correct seed to ensure consistent splits
#         ds_data = Dataset(ds, model=model,device=device, seed=seed)
#         ds_max_len = ds_data.max_len
#         ds_cache_dir = cache_dir / ds
#         logger.log(f"  - Ensuring activations for {ds} ({split} split): {ds_cache_dir} (max_len={ds_max_len})")
        
#         act_manager = ActivationManager(model, device, d_model=d_model, 
#                                         max_len=ds_max_len, cache_root=cache_dir)
        
#         # Get the correct split's text data
#         if split == 'train':
#             texts, _ = ds_data.get_train_set()
#         elif split == 'test':
#             texts, _ = ds_data.get_test_set()
#         else:
#             raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.")
            
#         arr = act_manager.get_activations(
#             texts, layer, component, use_cache=True, cache_dir=ds_cache_dir, logger=logger
#         )
#         if ds_max_len < final_max_len:
#             pad_width = ((0, 0), (0, final_max_len - ds_max_len), (0, 0))
#             arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
#         elif ds_max_len > final_max_len:
#             arr = arr[:, :final_max_len, :]
#         acts_list.append(np.copy(arr))
#     combined = np.concatenate(acts_list, axis=0)
#     return combined

def train_probe(
    model, d_model: int, train_dataset_name: str, layer: int, component: str,
    architecture_name: str, aggregation: str, config_name: str, device: str, use_cache: bool,
    seed: int, results_dir: Path, cache_dir: Path, logger: Logger, retrain: bool,
    test_size: float = 0.15,
):
    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, aggregation, layer, component)
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
    if use_cache and probe_state_path.exists() and not retrain:
        logger.log(f"  - Probe already trained. Skipping: {probe_state_path.name}")
        return
    logger.log("  - Training new probe …")

    train_ds = Dataset(train_dataset_name, model=model, device=device, seed=seed)  # uses default cache_root
    train_ds.split_data(test_size=test_size, seed=seed)  # Split the data
    train_acts, y_train = train_ds.get_train_set_activations(layer, component)

    probe = get_probe_architecture(architecture_name, d_model=d_model, device=device, aggregation=aggregation)
    fit_params = asdict(PROBE_CONFIGS[config_name])
    probe.fit(train_acts, y_train, **fit_params)

    probe_save_dir.mkdir(parents=True, exist_ok=True)
    probe.save_state(probe_state_path)
    logger.log(f"  - 🔥 Probe state saved to {probe_state_path.name}")

def evaluate_probe(
    train_dataset_name: str, eval_dataset_name: str, layer: int, component: str,
    architecture_config: dict, aggregation: str, results_dir: Path, logger: Logger,
    seed: int, model, d_model: int, device: str, use_cache: bool, cache_dir: Path, reevaluate: bool,
    test_size: float = 0.15, logit_diff_threshold: float = 2.5, score_options: list = None,
):
    if score_options is None:
        score_options = ['all']
    
    architecture_name = architecture_config["name"]
    config_name = architecture_config["config_name"]
    # For attention probes, we use "attention" as the aggregation name in results
    agg_name = "attention" if architecture_name == "attention" else aggregation

    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, aggregation, layer, component)
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
    eval_results_path = probe_save_dir / f"eval_on_{eval_dataset_name}__{probe_filename_base}_{agg_name}_results.json"

    if use_cache and eval_results_path.exists() and not reevaluate:
        logger.log("  - 😋 Using cached evaluation result ")
        return

    # Load probe
    probe = get_probe_architecture(architecture_name, d_model=d_model, device=device, aggregation=aggregation)
    probe.load_state(probe_state_path)

    # load activations via Dataset 
    eval_ds = Dataset(eval_dataset_name, model=model, device=device, seed=seed)
    eval_ds.split_data(test_size=test_size, seed=seed)  # Split the data
    test_acts, y_test = eval_ds.get_test_set_activations(layer, component)

    # Calculate metrics based on score options
    combined_metrics = {}
    
    if 'all' in score_options:
        logger.log(f"  - 📊 Calculating metrics for all examples...")
        all_metrics = probe.score(test_acts, y_test)
        all_metrics["filtered"] = False
        combined_metrics["all_examples"] = all_metrics
    
    if 'filtered' in score_options:
        logger.log(f"  - 🔍 Calculating filtered metrics (threshold={logit_diff_threshold})...")
        filtered_metrics = probe.score_filtered(test_acts, y_test, eval_dataset_name, results_dir, 
                                              seed=seed, logit_diff_threshold=logit_diff_threshold, 
                                              test_size=test_size)
        combined_metrics["filtered_examples"] = filtered_metrics
    
    # If only one option is specified, use that as the main result
    # if len(combined_metrics) == 1:
    #     main_key = list(combined_metrics.keys())[0]
    #     combined_metrics = combined_metrics[main_key]
    
    # Save metrics
    with open(eval_results_path, "w") as f:
        import json
        json.dump({"metrics": combined_metrics}, f, indent=2)
    
    # Log the results
    if 'all' in score_options:
        all_metrics = combined_metrics.get("all_examples", combined_metrics)
        logger.log(f"  - ❤️‍🔥 Success! All examples metrics: {all_metrics}")
    
    if 'filtered' in score_options:
        filtered_metrics = combined_metrics.get("filtered_examples", {})
        if filtered_metrics.get("filtered", False):
            logger.log(f"  - 🙃 Filtered examples metrics (threshold={logit_diff_threshold}): {filtered_metrics}")
        else:
            logger.log(f"  - 😵‍💫 Filtered scoring failed, using all examples")

