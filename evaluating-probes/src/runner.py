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
import numpy as np
import torch
import copy
import json
import pandas as pd


def get_activation_type_from_config(config_name: str) -> str:
    """
    Determine activation_type based on config_name.
    
    Args:
        config_name: Name of the probe configuration
        
    Returns:
        activation_type: Type of activations to use
    """
    if config_name == "attention":
        return "full"
    elif config_name.startswith("sklearn_linear"):
        # Extract aggregation method from config_name (e.g., "sklearn_linear_mean" -> "linear_mean")
        aggregation = config_name.split("_")[-1] if "_" in config_name else "mean"
        return f"linear_{aggregation}"
    elif config_name.startswith("linear"):
        # PyTorch linear probes use full activations with masks
        return "full"
    elif config_name.startswith("sae"):
        # SAE probes use pre-aggregated activations
        aggregation = config_name.split("_")[-1] if "_" in config_name else "mean"
        return f"sae_{aggregation}"
    elif config_name.startswith("act_sim"):
        # Activation similarity probes use pre-aggregated activations
        aggregation = config_name.split("_")[-1] if "_" in config_name else "mean"
        return f"act_sim_{aggregation}"
    else:
        # Default to full activations
        return "full"

def train_probe(
    model_name: str, d_model: int, train_dataset_name: str, layer: int, component: str,
    architecture_name: str, config_name: str, device: str, use_cache: bool,
    seed: int, results_dir: Path, cache_dir: Path, logger: Logger, retrain: bool,
    train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15,
    hyperparameter_tuning: bool = False,
    rebuild_config: dict = None,
    metric: str = 'acc',
    retrain_with_best_hparams: bool = False,
):
    # Only raise error if both hyperparameter_tuning and retrain_with_best_hparams are set
    if hyperparameter_tuning and retrain_with_best_hparams:
        raise ValueError("Cannot use both hyperparameter_tuning and retrain_with_best_hparams at the same time.")
    
    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name)
    probe_save_dir = results_dir / f"train_{train_dataset_name}"
    
    # If rebuilding, save in dataclass_exps_{dataset_name}
    if rebuild_config is not None:
        probe_save_dir = results_dir / f"dataclass_exps_{train_dataset_name}"
        probe_save_dir.mkdir(parents=True, exist_ok=True)
        suffix = rebuild_suffix(rebuild_config)
        probe_filename = f"{probe_filename_base}_{suffix}_state.npz"
        probe_state_path = probe_save_dir / probe_filename
        probe_json_path = probe_save_dir / f"{probe_filename_base}_{suffix}_meta.json"
        # For hyperparameter tuning, use the complete filename base with suffix
        hyperparam_filename_base = f"{probe_filename_base}_{suffix}"
    else:
        probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
        probe_json_path = probe_save_dir / f"{probe_filename_base}_meta.json"
        # For hyperparameter tuning, use the base filename
        hyperparam_filename_base = probe_filename_base
    
    # EARLY CHECK: If probe already exists and we're not retraining, skip immediately
    if use_cache and probe_state_path.exists() and not retrain:
        if rebuild_config is not None:
            logger.log(f"  - [SKIP] Probe already trained in dataclass_exps: {probe_state_path.name}")
        else:
            logger.log(f"  - [SKIP] Probe already trained: {probe_state_path.name}")
        return
    
    logger.log("  - Training new probe …")

    # Prepare dataset
    logger.log(f"  [DEBUG] Starting dataset preparation...")
    if rebuild_config is not None:
        logger.log(f"  [DEBUG] Using rebuild_config: {rebuild_config}")
        orig_ds = Dataset(train_dataset_name, model_name=model_name, device=device, seed=seed)
        logger.log(f"  [DEBUG] Created original dataset")
        
        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
            logger.log(f"  [DEBUG] Using LLM upsampling method")
            n_real_neg = rebuild_config.get('n_real_neg')
            n_real_pos = rebuild_config.get('n_real_pos')
            upsampling_factor = rebuild_config.get('upsampling_factor')
            
            run_name = str(results_dir).split('/')[-3]
            llm_csv_base_path = rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError("For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified")
            
            logger.log(f"  [DEBUG] Building LLM upsampled dataset...")
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
            logger.log(f"  [DEBUG] Built LLM upsampled dataset")
        else:
            # Original rebuild_config logic
            logger.log(f"  [DEBUG] Using original rebuild_config logic")
            train_class_counts = rebuild_config.get('class_counts')
            train_class_percents = rebuild_config.get('class_percents')
            train_total_samples = rebuild_config.get('total_samples')
            logger.log(f"  [DEBUG] Building imbalanced train balanced eval dataset...")
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
            logger.log(f"  [DEBUG] Built imbalanced train balanced eval dataset")
    else:
        logger.log(f"  [DEBUG] Using simple dataset creation")
        train_ds = Dataset(train_dataset_name, model_name=model_name, device=device, seed=seed)
        logger.log(f"  [DEBUG] Created dataset, now splitting data...")
        train_ds.split_data(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)
        logger.log(f"  [DEBUG] Split data complete")

    # Get activations
    activation_type = get_activation_type_from_config(config_name)
    logger.log(f"  [DEBUG] Using activation_type: {activation_type}")
    
    # Get activations using the unified method
    train_result = train_ds.get_train_set_activations(layer, component, activation_type=activation_type)
    
    # Handle different return formats (with/without masks)
    if len(train_result) == 3:
        train_acts, train_masks, y_train = train_result
    else:
        train_acts, y_train = train_result
        train_masks = None
    
    logger.log(f"  [DEBUG] Got activations with type: {activation_type}")

    print(f"Train activations: {len(train_acts) if isinstance(train_acts, list) else train_acts.shape}")

    # Create probe based on config_name
    logger.log(f"  [DEBUG] Creating probe for config_name: {config_name}")
    probe_config = asdict(PROBE_CONFIGS[config_name])
    logger.log(f"  [DEBUG] Got probe config")
    
    if config_name.startswith("sklearn_linear"):
        logger.log(f"  [DEBUG] Creating sklearn linear probe")
        # Sklearn linear probe
        probe = get_probe_architecture("sklearn_linear", d_model=d_model, device=device, config=probe_config)
        logger.log(f"  [DEBUG] Created sklearn linear probe")
        
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
            # Update probe parameters with best hyperparameters
            for key, value in best_params.items():
                setattr(probe, key, value)
            logger.log(f"  [DEBUG] Fitting probe with best hyperparameters...")
            # Pass masks only if they exist (for pre-aggregated activations, masks=None)
            if train_masks is not None:
                probe.fit(train_acts, y_train, train_masks)
            else:
                probe.fit(train_acts, y_train)
            logger.log(f"  [DEBUG] Fitted probe with best hyperparameters")
        elif hyperparameter_tuning:
            logger.log(f"  [DEBUG] Starting hyperparameter tuning...")
            # Sklearn probes don't use epochs, PyTorch probes do
            if config_name.startswith("sklearn_linear"):
                best_params = probe.find_best_fit(
                    train_acts, y_train,
                    verbose=True,
                    probe_save_dir=probe_save_dir, probe_filename_base=hyperparam_filename_base,
                    n_jobs=1  # Use single-threaded to avoid conflicts with main parallelization
                )
            else:
                # PyTorch-based probes use epochs
                best_params = probe.find_best_fit(
                    train_acts, y_train,
                    epochs=probe_config.get('epochs', 100),  # Get epochs from probe config
                    n_trials=20, direction=None, metric=metric,
                    probe_save_dir=probe_save_dir, probe_filename_base=hyperparam_filename_base,
                    n_jobs=1  # Use single-threaded to avoid conflicts with main parallelization
                )
            logger.log(f"  [DEBUG] Completed hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting probe normally...")
            # Pass masks only if they exist (for pre-aggregated activations, masks=None)
            if train_masks is not None:
                probe.fit(train_acts, y_train, train_masks)
            else:
                probe.fit(train_acts, y_train)
            logger.log(f"  [DEBUG] Fitted probe normally")
            
    elif config_name.startswith("linear"):
        logger.log(f"  [DEBUG] Creating PyTorch linear probe")
        # PyTorch linear probe
        probe = get_probe_architecture("linear", d_model=d_model, device=device, config=probe_config)
        logger.log(f"  [DEBUG] Created PyTorch linear probe")
        
        # Get fit parameters by filtering out initialization-only parameters
        fit_params = {}
        fit_param_names = ['lr', 'epochs', 'batch_size', 'weight_decay', 'verbose', 'early_stopping', 'patience', 'min_delta']
        for key, value in probe_config.items():
            if key in fit_param_names:
                fit_params[key] = value
        
        if retrain_with_best_hparams:
            logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
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
            logger.log(f"  [DEBUG] Fitting PyTorch linear probe with best hyperparameters...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted PyTorch linear probe with best hyperparameters")
        elif hyperparameter_tuning:
            logger.log(f"  [DEBUG] Starting PyTorch linear hyperparameter tuning...")
            best_params = probe.find_best_fit(
                train_acts, y_train,
                epochs=probe_config.get('epochs', 100),  # Get epochs from probe config
                n_trials=20, direction=None, metric=metric,
                probe_save_dir=probe_save_dir, probe_filename_base=hyperparam_filename_base
            )
            logger.log(f"  [DEBUG] Completed PyTorch linear hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting PyTorch linear probe normally...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted PyTorch linear probe normally")
            
    elif config_name == "attention":
        logger.log(f"  [DEBUG] Creating attention probe")
        # Attention probe
        probe = get_probe_architecture("attention", d_model=d_model, device=device, config=probe_config)
        logger.log(f"  [DEBUG] Created attention probe")
        
        # Get fit parameters
        fit_params = {}
        fit_param_names = ['lr', 'epochs', 'batch_size', 'weight_decay', 'verbose', 'early_stopping', 'patience', 'min_delta']
        for key, value in probe_config.items():
            if key in fit_param_names:
                fit_params[key] = value
        
        if retrain_with_best_hparams:
            logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
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
            logger.log(f"  [DEBUG] Fitting attention probe with best hyperparameters...")
            probe.fit(train_acts, y_train)
            logger.log(f"  [DEBUG] Fitted attention probe with best hyperparameters")
        elif hyperparameter_tuning:
            logger.log(f"  [DEBUG] Starting attention hyperparameter tuning...")
            best_params = probe.find_best_fit(
                train_acts, y_train,
                epochs=probe_config.get('epochs', 100),  # Get epochs from probe config
                n_trials=20, direction=None, verbose=True, metric=metric,
                probe_save_dir=probe_save_dir, probe_filename_base=hyperparam_filename_base
            )
            logger.log(f"  [DEBUG] Completed attention hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting attention probe normally...")
            probe.fit(train_acts, y_train)
            logger.log(f"  [DEBUG] Fitted attention probe normally")
            
    elif config_name.startswith("sae"):
        logger.log(f"  [DEBUG] Creating SAE probe")
        # SAE probe
        probe = get_probe_architecture("sae", d_model=d_model, device=device, config=probe_config)
        logger.log(f"  [DEBUG] Created SAE probe")
        
        # Get fit parameters
        fit_params = {}
        fit_param_names = ['lr', 'epochs', 'batch_size', 'weight_decay', 'verbose', 'early_stopping', 'patience', 'min_delta']
        for key, value in probe_config.items():
            if key in fit_param_names:
                fit_params[key] = value
        
        if retrain_with_best_hparams:
            logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
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
            logger.log(f"  [DEBUG] Fitting SAE probe with best hyperparameters...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted SAE probe with best hyperparameters")
        elif hyperparameter_tuning:
            logger.log(f"  [DEBUG] Starting SAE hyperparameter tuning...")
            best_params = probe.find_best_fit(
                train_acts, y_train,
                epochs=probe_config.get('epochs', 100),  # Get epochs from probe config
                n_trials=20, direction=None, metric=metric,
                probe_save_dir=probe_save_dir, probe_filename_base=hyperparam_filename_base
            )
            logger.log(f"  [DEBUG] Completed SAE hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting SAE probe normally...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted SAE probe normally")
            
    elif config_name == "mass_mean":
        logger.log(f"  [DEBUG] Creating mass mean probe")
        # Mass mean probe (non-trainable)
        probe = get_probe_architecture("mass_mean", d_model=d_model, device=device, config=probe_config)
        logger.log(f"  [DEBUG] Created mass mean probe")
        logger.log(f"  [DEBUG] Fitting mass mean probe...")
        probe.fit(train_acts, y_train)
        logger.log(f"  [DEBUG] Fitted mass mean probe")
        
    elif config_name.startswith("act_sim"):
        logger.log(f"  [DEBUG] Creating activation similarity probe")
        # Activation similarity probe (non-trainable)
        probe = get_probe_architecture("act_sim", d_model=d_model, device=device, config=probe_config)
        logger.log(f"  [DEBUG] Created activation similarity probe")
        logger.log(f"  [DEBUG] Fitting activation similarity probe...")
        probe.fit(train_acts, y_train)
        logger.log(f"  [DEBUG] Fitted activation similarity probe")
        
    else:
        raise ValueError(f"Unknown config_name: {config_name}")

    # Save probe
    logger.log(f"  [DEBUG] Saving probe state...")
    probe_save_dir.mkdir(parents=True, exist_ok=True)
    probe.save_state(probe_state_path)
    logger.log(f"  [DEBUG] Saved probe state")
    
    # Save metadata/config
    logger.log(f"  [DEBUG] Saving metadata...")
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
    logger.log(f"  [DEBUG] Saved metadata")
    logger.log(f"  - 🔥 Probe state saved to {probe_state_path.name}")
    logger.log(f"  [DEBUG] train_probe function completed successfully")


def evaluate_probe(
    train_dataset_name: str, eval_dataset_name: str, layer: int, component: str,
    architecture_config: dict, results_dir: Path, logger: Logger,
    seed: int, model_name: str, d_model: int, device: str, use_cache: bool, cache_dir: Path, reevaluate: bool,
    train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15, 
    score_options: list = None,
    rebuild_config: dict = None,
    return_metrics: bool = False,
):
    if score_options is None:
        score_options = ['all']
    
    architecture_name = architecture_config["name"]
    config_name = architecture_config["config_name"]
    # Extract aggregation from config for results, to be backward compatible
    agg_name = extract_aggregation_from_config(config_name, architecture_name)

    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name)
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
    
    if config_name.startswith("sklearn_linear"):
        probe = get_probe_architecture("sklearn_linear", d_model=d_model, device=device, config=probe_config)
    elif config_name.startswith("linear"):
        probe = get_probe_architecture("linear", d_model=d_model, device=device, config=probe_config)
    elif config_name == "attention":
        probe = get_probe_architecture("attention", d_model=d_model, device=device, config=probe_config)
    elif config_name.startswith("sae"):
        probe = get_probe_architecture("sae", d_model=d_model, device=device, config=probe_config)
    elif config_name == "mass_mean":
        probe = get_probe_architecture("mass_mean", d_model=d_model, device=device, config=probe_config)
    elif config_name.startswith("act_sim"):
        probe = get_probe_architecture("act_sim", d_model=d_model, device=device, config=probe_config)
    else:
        raise ValueError(f"Unknown config_name: {config_name}")
    
    probe.load_state(probe_state_path)
    
    # Get batch_size from probe config for evaluation
    # For SAE probes, use training_batch_size; for others, use batch_size
    if config_name.startswith("sae"):
        batch_size = probe_config.get('training_batch_size', probe_config.get('batch_size', 200))
    else:
        batch_size = probe_config.get('batch_size', 200)

    # Prepare evaluation dataset
    only_test = (eval_dataset_name != train_dataset_name)
    if rebuild_config is not None:
        orig_ds = Dataset(eval_dataset_name, model_name=model_name, device=device, seed=seed, only_test=only_test)
        
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
        eval_ds = Dataset(eval_dataset_name, model_name=model_name, device=device, seed=seed, only_test=only_test)
        eval_ds.split_data(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)
    
    # Get activations
    activation_type = get_activation_type_from_config(config_name)
    logger.log(f"  [DEBUG] Using activation_type: {activation_type}")
    
    # Get activations using the unified method
    test_result = eval_ds.get_test_set_activations(layer, component, activation_type=activation_type)
    
    # Handle different return formats (with/without masks)
    if len(test_result) == 3:
        test_acts, test_masks, y_test = test_result
    else:
        test_acts, y_test = test_result
        test_masks = None
    
    print(f"Test activations: {len(test_acts) if isinstance(test_acts, list) else test_acts.shape}")
    
    # Calculate metrics based on score options
    combined_metrics = {}
    
    if 'all' in score_options:
        logger.log(f"  - 🤗 Calculating all examples metrics...")
        if config_name == "attention" or config_name.startswith("act_sim"):
            # Attention probes and act_sim probes don't use masks
            all_metrics = probe.score(test_acts, y_test)
        else:
            # Other probes use masks
            all_metrics = probe.score(test_acts, y_test, masks=test_masks)
        combined_metrics["all_examples"] = all_metrics

    if 'filtered' in score_options:
        logger.log(f"  - 🤗 Calculating filtered metrics...")
        try:
            if config_name == "attention" or config_name.startswith("act_sim"):
                # Attention probes and act_sim probes don't use masks
                filtered_metrics = probe.score_filtered(test_acts, y_test, dataset_name=eval_dataset_name, 
                                                      results_dir=results_dir, seed=seed, test_size=test_size)
            else:
                # Other probes use masks
                filtered_metrics = probe.score_filtered(test_acts, y_test, masks=test_masks, 
                                                      dataset_name=eval_dataset_name, results_dir=results_dir, 
                                                      seed=seed, test_size=test_size)
            combined_metrics["filtered_examples"] = filtered_metrics
        except Exception as e:
            logger.log(f"  - 😵‍💫 Filtered scoring failed with error: {str(e)}")
            logger.log(f"  - 🔍 Debug info: eval_dataset_name={eval_dataset_name}, results_dir={results_dir}")
            # Set a default filtered metrics structure to indicate failure
            combined_metrics["filtered_examples"] = {
                "filtered": False,
                "error": str(e),
                "fallback_to_all": True
            }

    # Save metrics and per-datapoint scores/labels
    # Compute per-datapoint probe scores (logits) and labels
    if config_name == "attention" or config_name.startswith("act_sim"):
        # Attention probes and act_sim probes don't use masks
        test_scores = probe.predict_logits(test_acts)
    else:
        # Other probes use masks
        test_scores = probe.predict_logits(test_acts, masks=test_masks)
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
            logger.log(f"  - Found CSV file: {csv_files[0]}")
            df = pd.read_csv(csv_files[0]) # will only be one csv file in the directory.
            use_in_filtered = df['use_in_filtered_scoring'].values
            filtered_indices = np.where(use_in_filtered == 1)[0]
            
            if len(filtered_indices) == 0:
                logger.log(f"  - 😱 No examples passed the filter (all use_in_filtered_scoring values are 0)")
                logger.log(f"  - Filter statistics: total={len(use_in_filtered)}, passed={len(filtered_indices)}")
            else:
                logger.log(f"  - 💪🏼 Successfully filtered {len(filtered_indices)} examples from {len(test_scores)} total")
                output_dict["filtered_scores"] = {
                    "scores": [test_scores[i] for i in filtered_indices],
                    "labels": [test_labels[i] for i in filtered_indices],
                    "filtered": True,
                    "filter_method": "use_in_filtered_scoring",
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
            logger.log(f"  - 🙃 Filtered examples metrics: {filtered_metrics}")
        else:
            error_msg = filtered_metrics.get("error", "Unknown error")
            logger.log(f"  - 😵‍💫 Filtered scoring failed: {error_msg}")
            logger.log(f"  - 📊 Falling back to using all examples for evaluation")
    if return_metrics:
        return combined_metrics


def run_non_trainable_probe(
    model_name: str, d_model: int, train_dataset_name: str, layer: int, component: str,
    architecture_name: str, config_name: str, device: str, use_cache: bool,
    seed: int, results_dir: Path, cache_dir: Path, logger: Logger, retrain: bool,
    train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15,
    rebuild_config: dict = None,
    metric: str = 'acc',
):
    """
    Run non-trainable probes (like activation similarity and mass-mean probes).
    These probes don't require training but need to compute parameters from training data.
    """
    probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name)
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
        orig_ds = Dataset(train_dataset_name, model_name=model_name, device=device, seed=seed)
        
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
            
            run_name = str(results_dir).split('/')[-3]
            llm_csv_base_path = rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=val_size,
                test_size=test_size,
                llm_csv_base_path=llm_csv_base_path,
            )
        else:
            # Original rebuild_config logic
            train_class_counts = rebuild_config.get('class_counts')
            train_class_percents = rebuild_config.get('class_percents')
            train_total_samples = rebuild_config.get('total_samples')
            ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
            )
    else:
        ds = Dataset(train_dataset_name, model_name=model_name, device=device, seed=seed)
        ds.split_data(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)

    # Get activations
    activation_type = get_activation_type_from_config(config_name)
    logger.log(f"  [DEBUG] Using activation_type: {activation_type}")
    
    # Get activations using the unified method
    train_result = ds.get_train_set_activations(layer, component, activation_type=activation_type)
    test_result = ds.get_test_set_activations(layer, component, activation_type=activation_type)
    
    # Handle different return formats (with/without masks)
    if len(train_result) == 3:
        train_acts, train_masks, y_train = train_result
    else:
        train_acts, y_train = train_result
        train_masks = None
        
    if len(test_result) == 3:
        test_acts, test_masks, y_test = test_result
    else:
        test_acts, y_test = test_result
        test_masks = None
    
    print(f"Train activations: {len(train_acts) if isinstance(train_acts, list) else train_acts.shape}")
    print(f"Test activations: {len(test_acts) if isinstance(test_acts, list) else test_acts.shape}")
    
    # Fit the non-trainable probe
    probe_config = asdict(PROBE_CONFIGS[config_name])
    
    if config_name == "mass_mean":
        probe = get_probe_architecture("mass_mean", d_model=d_model, device=device, config=probe_config)
    elif config_name.startswith("act_sim"):
        probe = get_probe_architecture("act_sim", d_model=d_model, device=device, config=probe_config)
    else:
        raise ValueError(f"Unknown non-trainable config_name: {config_name}")
    
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

