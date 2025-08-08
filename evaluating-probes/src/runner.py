from pathlib import Path
from dataclasses import asdict
from typing import Any
from src.data import Dataset
from src.logger import Logger
from configs.probes import PROBE_CONFIGS
from src.utils import (
    extract_aggregation_from_config,
    get_probe_architecture,
    get_probe_filename_prefix,
    rebuild_suffix
)
from src.probes import GroupedProbe
import numpy as np
import torch
import copy
import json
import pandas as pd

def train_probe(
    model, d_model: int, train_dataset_name: str, layer: int, component: str,
    architecture_name: str, config_name: str, device: str, use_cache: bool,
    seed: int, results_dir: Path, cache_dir: Path, logger: Logger, retrain: bool,
    train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15,
    hyperparameter_tuning: bool = False,
    rebuild_config: dict = None,
    # return_probe_and_test: bool = False,
    metric: str = 'acc',
    retrain_with_best_hparams: bool = False,
    contrast_fn: Any = None,
):
    # Only raise error if both hyperparameter_tuning and retrain_with_best_hparams are set
    if hyperparameter_tuning and retrain_with_best_hparams:
        raise ValueError("Cannot use both hyperparameter_tuning and retrain_with_best_hparams at the same time.")
    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name, contrast_fn)
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    # If rebuilding, save in dataclass_exps_{dataset_name}
    if rebuild_config is not None:
        probe_save_dir = results_dir / f"dataclass_exps_{train_dataset_name}"
        probe_save_dir.mkdir(parents=True, exist_ok=True)
        suffix = rebuild_suffix(rebuild_config)
        probe_filename = f"{probe_filename_base}_{suffix}_state.npz"
        probe_state_path = probe_save_dir / probe_filename
        probe_json_path = probe_save_dir / f"{probe_filename_base}_{suffix}_meta.json"
        # Check for existing probe file before running
        if use_cache and probe_state_path.exists() and not retrain:
            logger.log(f"  - [SKIP] Probe already trained in dataclass_exps: {probe_state_path.name}")
            return
    else:
        probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
        probe_json_path = probe_save_dir / f"{probe_filename_base}_meta.json"
    if use_cache and probe_state_path.exists() and not retrain:
        logger.log(f"  - Probe already trained. Skipping: {probe_state_path.name}")
        return
    logger.log("  - Training new probe …")

    if rebuild_config is not None:
        orig_ds = Dataset(train_dataset_name, model=model, device=device, seed=seed)
        
        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
            n_real_neg = rebuild_config.get('n_real_neg')
            n_real_pos = rebuild_config.get('n_real_pos')
            upsampling_factor = rebuild_config.get('upsampling_factor')
            
            run_name = str(results_dir).split('/')[-3]
            llm_csv_base_path = rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError("For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified")
            
            train_ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=val_size,
                test_size=test_size,
                llm_csv_base_path=llm_csv_base_path,
                only_test=False,
            )
        else:
            # Original rebuild_config logic
            train_class_counts = rebuild_config.get('class_counts')
            train_class_percents = rebuild_config.get('class_percents')
            train_total_samples = rebuild_config.get('total_samples')
            train_ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=val_size,
                test_size=test_size,
                seed=seed,  # Use the global seed passed to this function
                only_test=False,
            )
    else:
        train_ds = Dataset(train_dataset_name, model=model, device=device, seed=seed)
        train_ds.split_data(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)

    # Use contrast activations if contrast_fn is provided
    if contrast_fn is not None:
        train_acts, y_train = train_ds.get_contrast_activations(contrast_fn, layer, component, split='train')
        val_acts, y_val = train_ds.get_contrast_activations(contrast_fn, layer, component, split='val')
        # test_acts, y_test = train_ds.get_contrast_activations(contrast_fn, layer, component, split='test')
    else:
        train_acts, y_train = train_ds.get_train_set_activations(layer, component)
        val_acts, y_val = train_ds.get_val_set_activations(layer, component)
        # test_acts, y_test = train_ds.get_test_set_activations(layer, component)

    probe = get_probe_architecture(architecture_name, d_model=d_model, device=device, config=asdict(PROBE_CONFIGS[config_name]))
    
    # Get fit parameters by filtering out initialization-only parameters
    all_params = asdict(PROBE_CONFIGS[config_name])
    fit_params = {}
    
    # Parameters that should be passed to fit() method
    fit_param_names = ['lr', 'epochs', 'batch_size', 'weight_decay', 'verbose', 'early_stopping', 'patience', 'min_delta']
    for key, value in all_params.items():
        if key in fit_param_names:
            fit_params[key] = value
    
    # Handle mass-mean probe configuration specially (binary classification only)
    if architecture_name in ["mass_mean", "mass_mean_iid"]:
        # Mass-mean probes don't have weighting_method, they're computed analytically
        weighting_method = "mass_mean"
    else:
        weighting_method = all_params.get("weighting_method", "weighted_sampler")

    if retrain_with_best_hparams:
        logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
        # Load best hyperparameters from the dedicated best_hparams.json file
        if rebuild_config is not None:
            suffix = rebuild_suffix(rebuild_config)
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_{suffix}_best_hparams.json"
        else:
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
        if not best_hparams_path.exists():
            raise FileNotFoundError(f"Best hyperparameters file not found: {best_hparams_path}")
        with open(best_hparams_path, 'r') as f:
            best_params = json.load(f)
        fit_params.update(best_params)
        if weighting_method == "mass_mean":
            # Mass-mean probe is computed analytically, no training needed
            logger.log(f"  - Computing mass-mean probe analytically...")
            probe.fit(train_acts, y_train, **fit_params)
        elif weighting_method == "weighted_loss":
            fit_params["use_weighted_loss"] = True
            fit_params["use_weighted_sampler"] = False
            probe.fit(train_acts, y_train, **fit_params)
        elif weighting_method == "weighted_sampler":
            fit_params["use_weighted_loss"] = False
            fit_params["use_weighted_sampler"] = True
            probe.fit(train_acts, y_train, **fit_params)
        else:
            raise ValueError(f"Unknown weighting_method: {weighting_method}")
    elif hyperparameter_tuning:
        best_params = probe.find_best_fit(
            train_acts, y_train, val_acts, y_val,
            mask_train=None, mask_val=None,
            n_trials=10, direction=None, verbose=True, weighting_method=weighting_method, metric=metric,
            probe_save_dir=probe_save_dir, probe_filename_base=probe_filename_base
        )
    else:
        if weighting_method == "mass_mean":
            # Mass-mean probe is computed analytically, no training needed
            logger.log(f"  - Computing mass-mean probe analytically...")
            probe.fit(train_acts, y_train, **fit_params)
        elif weighting_method == "weighted_loss":
            fit_params["use_weighted_loss"] = True
            fit_params["use_weighted_sampler"] = False
            probe.fit(train_acts, y_train, **fit_params)
        elif weighting_method == "weighted_sampler":
            fit_params["use_weighted_loss"] = False
            fit_params["use_weighted_sampler"] = True
            probe.fit(train_acts, y_train, **fit_params)
        else:
            raise ValueError(f"Unknown weighting_method: {weighting_method}")

    probe_save_dir.mkdir(parents=True, exist_ok=True)
    probe.save_state(probe_state_path)
    # Save metadata/config
    meta = {
        'train_dataset_name': train_dataset_name,
        'layer': layer,
        'component': component,
        'architecture_name': architecture_name,
        'aggregation': extract_aggregation_from_config(config_name, architecture_name) if hasattr(probe, 'aggregation') else None,
        'config_name': config_name,
        'rebuild_config': copy.deepcopy(rebuild_config),
        'probe_state_path': str(probe_state_path),
    }
    if hyperparameter_tuning:
        meta['hyperparameters'] = best_params
    with open(probe_json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.log(f"  - 🔥 Probe state saved to {probe_state_path.name}")
    # if return_probe_and_test:
    #     return probe, test_acts, y_test


def train_grouped_probe(
    model, d_model: int, train_dataset_name: str, layer: int, component: str,
    architecture_configs: list, device: str, use_cache: bool,
    seed: int, results_dir: Path, cache_dir: Path, logger: Logger, retrain: bool,
    train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15,
    rebuild_config: dict = None,
    metric: str = 'acc',
    contrast_fn: Any = None,
):
    """
    Train multiple probes simultaneously using GroupedProbe to reduce GPU memory transfers.
    Each probe is saved individually with its own filename for compatibility with evaluation code.
    
    Args:
        architecture_configs: List of dicts with 'name' and 'config_name' keys
    """
    logger.log(f"  - Training {len(architecture_configs)} probes using grouped training for efficiency...")

    # Check which individual probes already exist and skip them
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    if rebuild_config is not None:
        probe_save_dir = results_dir / f"dataclass_exps_{train_dataset_name}"
    
    probes_to_train = []
    existing_probes = []
    
    for arch_config in architecture_configs:
        architecture_name = arch_config['name']
        config_name = arch_config.get('config_name')
        
        individual_probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name, contrast_fn)
        
        if rebuild_config is not None:
            suffix = rebuild_suffix(rebuild_config)
            individual_probe_state_path = probe_save_dir / f"{individual_probe_filename_base}_{suffix}_state.npz"
        else:
            individual_probe_state_path = probe_save_dir / f"{individual_probe_filename_base}_state.npz"
        
        if use_cache and individual_probe_state_path.exists() and not retrain:
            existing_probes.append(arch_config)
            logger.log(f"    - [SKIP] Probe already exists: {architecture_name} ({config_name})")
        else:
            probes_to_train.append(arch_config)
    
    if not probes_to_train:
        logger.log(f"  - [SKIP] All {len(architecture_configs)} probes already trained. Skipping grouped training.")
        return
        
    logger.log(f"  - Training {len(probes_to_train)} probes (skipping {len(existing_probes)} existing probes)")
    
    # Keep original list for metadata saving, update working list for training
    original_architecture_configs = architecture_configs
    architecture_configs = probes_to_train

    # Prepare dataset (same logic as train_probe)
    if rebuild_config is not None:
        orig_ds = Dataset(train_dataset_name, model=model, device=device, seed=seed)
        
        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
            n_real_neg = rebuild_config.get('n_real_neg')
            n_real_pos = rebuild_config.get('n_real_pos')
            upsampling_factor = rebuild_config.get('upsampling_factor')
            
            run_name = str(results_dir).split('/')[-3]
            llm_csv_base_path = rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError("For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified")
            
            train_ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=val_size,
                test_size=test_size,
                llm_csv_base_path=llm_csv_base_path,
                only_test=False,
            )
        else:
            # Original rebuild_config logic
            train_class_counts = rebuild_config.get('class_counts')
            train_class_percents = rebuild_config.get('class_percents')
            train_total_samples = rebuild_config.get('total_samples')
            train_ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                only_test=False,
            )
    else:
        train_ds = Dataset(train_dataset_name, model=model, device=device, seed=seed)
        train_ds.split_data(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)

    # Use contrast activations if contrast_fn is provided
    if contrast_fn is not None:
        train_acts, y_train = train_ds.get_contrast_activations(contrast_fn, layer, component, split='train')
        val_acts, y_val = train_ds.get_contrast_activations(contrast_fn, layer, component, split='val')
    else:
        train_acts, y_train = train_ds.get_train_set_activations(layer, component)
        val_acts, y_val = train_ds.get_val_set_activations(layer, component)

    # Create grouped probe
    grouped_probe = GroupedProbe(d_model=d_model, device=device)
    
    # Add each probe to the group
    for i, arch_config in enumerate(architecture_configs):
        architecture_name = arch_config['name']
        config_name = arch_config.get('config_name')
        
        # Create a unique name for this probe within the group
        probe_name = f"{architecture_name}_{config_name}_{i}"
        
        # Skip non-trainable probes and SAE probes (they train faster individually)
        if (architecture_name.startswith('act_sim') or 
            architecture_name in ['mass_mean', 'mass_mean_iid'] or
            architecture_name.startswith('sae')):
            logger.log(f"    - Skipping probe (non-trainable or SAE): {architecture_name}, {config_name}")
            continue
            
        grouped_probe.add_probe(probe_name, architecture_name, config_name)
    
    if not grouped_probe.probes:
        logger.log("  - No trainable probes to group. Skipping grouped training.")
        return
    
    # Get fit parameters from the first probe's config (they should be similar)
    first_probe_name = list(grouped_probe.probes.keys())[0]
    all_params = grouped_probe.probe_configs[first_probe_name]
    fit_params = {}
    
    # Parameters that should be passed to fit() method
    fit_param_names = ['lr', 'epochs', 'batch_size', 'weight_decay', 'verbose', 'early_stopping', 'patience', 'min_delta']
    for key, value in all_params.items():
        if key in fit_param_names:
            fit_params[key] = value
    
    # Handle weighting method
    weighting_method = all_params.get("weighting_method", "weighted_sampler")
    
    if weighting_method == "weighted_loss":
        fit_params["use_weighted_loss"] = True
        fit_params["use_weighted_sampler"] = False
    elif weighting_method == "weighted_sampler":
        fit_params["use_weighted_loss"] = False
        fit_params["use_weighted_sampler"] = True
    else:
        fit_params["use_weighted_loss"] = False
        fit_params["use_weighted_sampler"] = False
    
    # Train the grouped probe
    logger.log(f"  - Training {len(grouped_probe.probes)} probes simultaneously...")
    grouped_probe.fit(train_acts, y_train, **fit_params)
    
    # Save each probe individually with its own filename (for compatibility with evaluation)
    probe_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the enhanced save_state method to save each probe with proper individual filenames
    grouped_probe.save_state(
        path=probe_save_dir / "grouped_probe_state.json",  # Dummy path, not actually used
        architecture_configs=architecture_configs,
        train_dataset_name=train_dataset_name,
        layer=layer,
        component=component,
        rebuild_config=rebuild_config,
        contrast_fn=contrast_fn
    )
    
    # Save metadata files for each probe (both trained and existing ones)
    for arch_config in original_architecture_configs:
        architecture_name = arch_config['name']
        config_name = arch_config.get('config_name')
        
        # Create individual probe filename (same as train_probe would use)
        individual_probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name, contrast_fn)
        
        if rebuild_config is not None:
            suffix = rebuild_suffix(rebuild_config)
            individual_probe_state_path = probe_save_dir / f"{individual_probe_filename_base}_{suffix}_state.npz"
            individual_probe_json_path = probe_save_dir / f"{individual_probe_filename_base}_{suffix}_meta.json"
        else:
            individual_probe_state_path = probe_save_dir / f"{individual_probe_filename_base}_state.npz"
            individual_probe_json_path = probe_save_dir / f"{individual_probe_filename_base}_meta.json"
        
        # Save metadata/config (same as train_probe would save)
        meta = {
            'train_dataset_name': train_dataset_name,
            'layer': layer,
            'component': component,
            'architecture_name': architecture_name,
            'aggregation': extract_aggregation_from_config(config_name, architecture_name),
            'config_name': config_name,
            'rebuild_config': copy.deepcopy(rebuild_config),
            'probe_state_path': str(individual_probe_state_path),
        }
        
        with open(individual_probe_json_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.log(f"  - 🔥 Saved individual probe: {individual_probe_state_path.name}")
    
    logger.log(f"  - ✅ Trained {len(grouped_probe.probes)} probes, metadata saved for all {len(original_architecture_configs)} probes")
    
    return grouped_probe


def evaluate_probe(
    train_dataset_name: str, eval_dataset_name: str, layer: int, component: str,
    architecture_config: dict, results_dir: Path, logger: Logger,
    seed: int, model, d_model: int, device: str, use_cache: bool, cache_dir: Path, reevaluate: bool,
    train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15, 
    logit_diff_threshold: float = 4, score_options: list = None,
    rebuild_config: dict = None,
    return_metrics: bool = False,
    contrast_fn: Any = None,  # NEW
):
    if score_options is None:
        score_options = ['all']
    
    architecture_name = architecture_config["name"]
    config_name = architecture_config["config_name"]
    # Extract aggregation from config for results, to be backward compatible
    agg_name = extract_aggregation_from_config(config_name, architecture_name)

    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name, contrast_fn)
    if rebuild_config is not None:
        probe_save_dir = results_dir / f"dataclass_exps_{train_dataset_name}"
        suffix = rebuild_suffix(rebuild_config)
        probe_state_path = probe_save_dir / f"{probe_filename_base}_{suffix}_state.npz"
        eval_results_path = probe_save_dir / f"eval_on_{eval_dataset_name}__{probe_filename_base}_{suffix}_seed{seed}_{agg_name}_results.json"
    else:
        probe_save_dir = results_dir / f"train_{train_dataset_name}"
        probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
        eval_results_path = probe_save_dir / f"eval_on_{eval_dataset_name}__{probe_filename_base}_seed{seed}_{agg_name}_results.json"

    if use_cache and eval_results_path.exists() and not reevaluate:
        logger.log("  - 😋 Using cached evaluation result ")
        if return_metrics:
            with open(eval_results_path, "r") as f:
                return json.load(f)["metrics"]
        return

    # Load probe
    probe_config = asdict(PROBE_CONFIGS[config_name])
    probe = get_probe_architecture(architecture_name, d_model=d_model, device=device, config=probe_config)
    probe.load_state(probe_state_path)
    
    # Get batch_size from probe config for evaluation
    # For SAE probes, use training_batch_size; for others, use batch_size
    if architecture_name.startswith("sae"):
        batch_size = probe_config.get('training_batch_size', probe_config.get('batch_size', 200))
    else:
        batch_size = probe_config.get('batch_size', 200)

    # Prepare evaluation dataset
    only_test = (eval_dataset_name != train_dataset_name)
    if rebuild_config is not None:
        orig_ds = Dataset(eval_dataset_name, model=model, device=device, seed=seed, only_test=only_test)
        
        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
            n_real_neg = rebuild_config.get('n_real_neg')
            n_real_pos = rebuild_config.get('n_real_pos')
            upsampling_factor = rebuild_config.get('upsampling_factor')
            
            run_name = str(results_dir).split('/')[-3]
            llm_csv_base_path = rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError("For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified")
            
            eval_ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=val_size,
                test_size=test_size,
                llm_csv_base_path=llm_csv_base_path,
                only_test=only_test,
            )
        else:
            # Original rebuild_config logic
            train_class_counts = rebuild_config.get('class_counts')
            train_class_percents = rebuild_config.get('class_percents')
            train_total_samples = rebuild_config.get('total_samples')
            eval_ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                only_test=only_test,
            )
    else:
        eval_ds = Dataset(eval_dataset_name, model=model, device=device, seed=seed, only_test=only_test)
        eval_ds.split_data(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)
    # Use contrast activations if contrast_fn is provided
    if contrast_fn is not None:
        test_acts, y_test = eval_ds.get_contrast_activations(contrast_fn, layer, component, split='test')
    else:
        test_acts, y_test = eval_ds.get_test_set_activations(layer, component)

    # Calculate metrics based on score options
    combined_metrics = {}

    if 'all' in score_options:
        logger.log(f"  - 🥰 Calculating metrics for all examples...")
        all_metrics = probe.score(test_acts, y_test, batch_size=batch_size)
        all_metrics["filtered"] = False
        combined_metrics["all_examples"] = all_metrics
    
    if 'filtered' in score_options:
        logger.log(f"  - 🤗 Calculating filtered metrics (threshold={logit_diff_threshold})...")
        filtered_metrics = probe.score_filtered(test_acts, y_test, eval_dataset_name, results_dir, 
                                              seed=seed, logit_diff_threshold=logit_diff_threshold, 
                                              test_size=test_size)
        combined_metrics["filtered_examples"] = filtered_metrics
    
    # Save metrics and per-datapoint scores/labels
    # Compute per-datapoint probe scores (logits) and labels
    test_scores = probe.predict_logits(test_acts, batch_size=batch_size)
    # Flatten if shape is (N, 1)
    if hasattr(test_scores, 'shape') and len(test_scores.shape) == 2 and test_scores.shape[1] == 1:
        test_scores = test_scores[:, 0]
    test_scores = test_scores.tolist()
    test_labels = y_test.tolist()

    # Build output structure - always save all scores in main scores field
    output_dict = {
        "metrics": combined_metrics,
        "scores": {
            "scores": test_scores,
            "labels": test_labels,
            "filtered": False
        }
    }

    # Add filtered scores only if filtering was requested and actually removed data points
    if 'filtered' in score_options and 'filtered_examples' in combined_metrics:
        filtered_metrics = combined_metrics["filtered_examples"]
        if filtered_metrics.get("filtered", False) and filtered_metrics.get("removed_count", 0) > 0:
            # Get filtered indices from the CSV file
            # Runthrough directory is always in the parent directory (results/{experiment_name}/)
            parent_dir = results_dir.parent
            runthrough_dir = parent_dir / f"runthrough_{eval_dataset_name}"
            
            csv_files = list(runthrough_dir.glob("*logit_diff*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                if 'logit_diff' in df.columns:
                    mask_filter = np.abs(df['logit_diff'].values) > logit_diff_threshold
                    filtered_indices = np.where(mask_filter)[0]
                    
                    if len(filtered_indices) > 0:
                        output_dict["filtered_scores"] = {
                            "scores": [test_scores[i] for i in filtered_indices],
                            "labels": [test_labels[i] for i in filtered_indices],
                            "filtered": True,
                            "logit_diff_threshold": logit_diff_threshold,
                            "original_size": len(test_scores),
                            "filtered_size": len(filtered_indices),
                            "removed_count": len(test_scores) - len(filtered_indices)
                        }
    
    with open(eval_results_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    
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
    if return_metrics:
        return combined_metrics


def run_non_trainable_probe(
    model, d_model: int, train_dataset_name: str, layer: int, component: str,
    architecture_name: str, config_name: str, device: str, use_cache: bool,
    seed: int, results_dir: Path, cache_dir: Path, logger: Logger, retrain: bool,
    train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15,
    rebuild_config: dict = None,
    metric: str = 'acc',
    contrast_fn: Any = None,
):
    """
    Run non-trainable probes (like activation similarity and mass-mean probes).
    These probes don't require training but need to compute parameters from training data.
    """
    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name, contrast_fn)
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    
    # If rebuilding, save in dataclass_exps_{dataset_name}
    if rebuild_config is not None:
        probe_save_dir = results_dir / f"dataclass_exps_{train_dataset_name}"
        probe_save_dir.mkdir(parents=True, exist_ok=True)
        suffix = rebuild_suffix(rebuild_config)
        probe_filename = f"{probe_filename_base}_{suffix}_state.npz"
        probe_state_path = probe_save_dir / probe_filename
        probe_json_path = probe_save_dir / f"{probe_filename_base}_{suffix}_meta.json"
        # Check for existing probe file before running
        if use_cache and probe_state_path.exists() and not retrain:
            logger.log(f"  - [SKIP] Non-trainable probe already computed in dataclass_exps: {probe_state_path.name}")
            return
    else:
        probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
        probe_json_path = probe_save_dir / f"{probe_filename_base}_meta.json"
    
    if use_cache and probe_state_path.exists() and not retrain:
        logger.log(f"  - Non-trainable probe already computed. Skipping: {probe_state_path.name}")
        return
    
    logger.log("  - Computing non-trainable probe parameters...")

    # Prepare dataset
    if rebuild_config is not None:
        orig_ds = Dataset(train_dataset_name, model=model, device=device, seed=seed)
        
        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
            n_real_neg = rebuild_config.get('n_real_neg')
            n_real_pos = rebuild_config.get('n_real_pos')
            upsampling_factor = rebuild_config.get('upsampling_factor')
            llm_seed = rebuild_config.get('seed')
            
            # Check if LLM upsampling seed matches the global seed to prevent duplicates
            if llm_seed is not None and llm_seed != seed:
                logger.log(f"  - [SKIP] LLM upsampling experiment seed ({llm_seed}) doesn't match global seed ({seed}). Skipping to prevent duplicates.")
                return
            
            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError("For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified")
            
            # Use run_name from config for LLM samples path
            # We need to get the run_name from the results_dir path: results/run_name/seed_X/experiment_name
            run_name = results_dir.parent.parent.name  # Go up two levels to get run_name
            llm_csv_base_path = rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            train_ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=val_size,
                test_size=test_size,
                llm_csv_base_path=llm_csv_base_path,
                only_test=False,
            )
        else:
            # Original rebuild_config logic
            train_class_counts = rebuild_config.get('class_counts')
            train_class_percents = rebuild_config.get('class_percents')
            train_total_samples = rebuild_config.get('total_samples')
            train_ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                only_test=False,
            )
    else:
        train_ds = Dataset(train_dataset_name, model=model, device=device, seed=seed)
        train_ds.split_data(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)

    # Get training activations
    if contrast_fn is not None:
        train_acts, y_train = train_ds.get_contrast_activations(contrast_fn, layer, component, split='train')
    else:
        train_acts, y_train = train_ds.get_train_set_activations(layer, component)

    # Create and fit the non-trainable probe
    probe_config = asdict(PROBE_CONFIGS[config_name])
    probe = get_probe_architecture(architecture_name, d_model=d_model, device=device, config=probe_config)
    
    # Batch size is already set during probe initialization
    probe.fit(train_acts, y_train)

    # Save the probe
    probe_save_dir.mkdir(parents=True, exist_ok=True)
    probe.save_state(probe_state_path)
    
    # Save metadata/config
    meta = {
        'train_dataset_name': train_dataset_name,
        'layer': layer,
        'component': component,
        'architecture_name': architecture_name,
        'aggregation': extract_aggregation_from_config(config_name, architecture_name) if hasattr(probe, 'aggregation') else None,
        'config_name': config_name,
        'rebuild_config': copy.deepcopy(rebuild_config),
        'probe_state_path': str(probe_state_path),
        'probe_type': 'non_trainable'
    }
    
    with open(probe_json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.log(f"  - 🔥 Non-trainable probe computed and saved to {probe_state_path.name}")

