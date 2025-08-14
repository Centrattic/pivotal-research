from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import copy
import json
import pandas as pd
from src.data import Dataset
from src.logger import Logger
from configs.probes import PROBE_CONFIGS
from src.utils import (
    extract_aggregation_from_config,
    get_probe_architecture,
    get_probe_filename_prefix,
    rebuild_suffix
)
from src.probe_configs import ProbeJob, LinearProbeConfig, SAEProbeConfig, AttentionProbeConfig, NonTrainableProbeConfig

def add_hyperparams_to_filename(base_filename: str, probe_config) -> str:
    """Add hyperparameter values to filename if they differ from defaults."""
    hparam_suffix = ""
    
    # Common hyperparameters to include
    hparams_to_check = ['C', 'lr', 'weight_decay', 'batch_size', 'epochs']
    
    for hparam in hparams_to_check:
        if hasattr(probe_config, hparam):
            value = getattr(probe_config, hparam)
            # Only add if it's not the default value
            if hparam == 'C' and value != 1.0:
                hparam_suffix += f"_C{value}"
            elif hparam == 'lr' and value != 5e-4:
                hparam_suffix += f"_lr{value}"
            elif hparam == 'weight_decay' and value != 0.0:
                hparam_suffix += f"_wd{value}"
            elif hparam == 'batch_size' and value != 1024:
                hparam_suffix += f"_bs{value}"
            elif hparam == 'epochs' and value != 100:
                hparam_suffix += f"_ep{value}"
    
    if hparam_suffix:
        return base_filename + hparam_suffix
    else:
        return base_filename

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

def extract_activations_for_dataset(model, tokenizer, model_name, dataset_name, layer, component, device, seed, logger, on_policy=False, include_llm_samples=True, format_type="r-no-it"):
    """
    Extract activations for a dataset using the full model.
    This is phase 1: ensure all activations are cached.
    Also pre-compute aggregated activations for faster loading later.
    If include_llm_samples=True, also extract activations for any LLM upsampled samples found.
    """
    logger.log(f"  - Extracting activations for {dataset_name}, L{layer}, {component} (on_policy={on_policy})")
    
    try:
        # Create dataset with full model to extract activations
        ds = Dataset(dataset_name, model=model, tokenizer=tokenizer, model_name=model_name, device=device, seed=seed)
        
        if on_policy:
            # For on-policy: extract activations using the specified format
            logger.log(f"    - On-policy: extracting activations using format '{format_type}'...")
            if hasattr(ds, 'response_texts') and ds.response_texts is not None:
                dataset_texts = ds.X.tolist()
                response_texts = ds.response_texts.tolist()
                newly_added = ds.act_manager.ensure_texts_cached(
                    dataset_texts, layer, component, 
                    response_texts=response_texts, format_type=format_type
                )
                logger.log(f"    - Cached {newly_added} new activations (on-policy format: {format_type})")
            else:
                logger.log(f"    - Warning: No response texts found for on-policy dataset")
                dataset_texts = ds.X.tolist()
                newly_added = ds.act_manager.ensure_texts_cached(dataset_texts, layer, component)
                logger.log(f"    - Cached {newly_added} new activations (dataset only)")
        else:
            # For off-policy: extract activations from prompts only
            logger.log(f"    - Off-policy: extracting activations from prompts...")
            dataset_texts = ds.X.tolist()
            newly_added = ds.act_manager.ensure_texts_cached(dataset_texts, layer, component)
            logger.log(f"    - Cached {newly_added} new activations (dataset)")

        # If including LLM samples, scan results/<run_name>/seed_*/llm_samples/samples_*.csv for prompts and cache them
        llm_samples_added = 0
        if include_llm_samples:
            try:
                results_dir = Path('results')
                llm_texts = []
                for run_dir in results_dir.glob('*'):
                    for seed_dir in run_dir.glob('seed_*'):
                        llm_dir = seed_dir / 'llm_samples'
                        if not llm_dir.exists(): # like for eval datasets, ex. 87_is_spam
                            continue
                        for csv_file in llm_dir.glob('samples_*.csv'):
                            try:
                                df = pd.read_csv(csv_file)
                                if 'prompt' in df.columns:
                                    llm_texts.extend(df['prompt'].astype(str).tolist())
                            except Exception:
                                continue
                if llm_texts:
                    llm_samples_added = ds.act_manager.ensure_texts_cached(llm_texts, layer, component)
                    logger.log(f"    - Cached {llm_samples_added} new activations (LLM)")
            except Exception as e:
                logger.log(f"    - ⚠️ LLM sample caching skipped due to error: {e}")

        # Then compute and save all aggregated activations (force if any new added)
        logger.log(f"    - Computing aggregated activations...")
        ds.act_manager.compute_and_save_all_aggregations(layer, component, force_recompute=(newly_added > 0 or llm_samples_added > 0))
        logger.log(f"    - Completed aggregations")

    except Exception as e:
        logger.log(f"    - 💀 ERROR extracting activations for {dataset_name}: {e}")
        raise

def train_single_probe(job: ProbeJob, config: Dict, results_dir: Path, cache_dir: Path, logger: Logger, retrain: bool, hyperparameter_tuning: bool, retrain_with_best_hparams: bool):
    """
    Train a single probe with the given configuration.
    """
    # Only raise error if both hyperparameter_tuning and retrain_with_best_hparams are set
    if hyperparameter_tuning and retrain_with_best_hparams:
        raise ValueError("Cannot use both hyperparameter_tuning and retrain_with_best_hparams at the same time.")
    
    # Create config name from architecture and probe config
    config_name = f"{job.architecture_name}_{job.probe_config.aggregation}" if hasattr(job.probe_config, 'aggregation') else job.architecture_name
    
    probe_filename_base = get_probe_filename_prefix(job.train_dataset, job.architecture_name, job.layer, job.component, config_name)
    probe_save_dir = results_dir / f"train_{job.train_dataset}"
    
    # Add hyperparameters to filename if they differ from defaults
    probe_filename_with_hparams = add_hyperparams_to_filename(probe_filename_base, job.probe_config)
    
    # If rebuilding, save in dataclass_exps_{dataset_name}
    if job.rebuild_config is not None:
        probe_save_dir = results_dir / f"dataclass_exps_{job.train_dataset}"
        probe_save_dir.mkdir(parents=True, exist_ok=True)
        suffix = rebuild_suffix(job.rebuild_config)
        probe_filename = f"{probe_filename_with_hparams}_{suffix}_state.npz"
        probe_state_path = probe_save_dir / probe_filename
        probe_json_path = probe_save_dir / f"{probe_filename_with_hparams}_{suffix}_meta.json"
        # For hyperparameter tuning, use the complete filename base with suffix
        hyperparam_filename_base = f"{probe_filename_with_hparams}_{suffix}"
    else:
        probe_state_path = probe_save_dir / f"{probe_filename_with_hparams}_state.npz"
        probe_json_path = probe_save_dir / f"{probe_filename_with_hparams}_meta.json"
        # For hyperparameter tuning, use the base filename
        hyperparam_filename_base = probe_filename_with_hparams
    
    # EARLY CHECK: If probe already exists and we're not retraining, skip immediately
    if config.get('cache_activations', True) and probe_state_path.exists() and not retrain:
        if job.rebuild_config is not None:
            logger.log(f"  - [SKIP] Probe already trained in dataclass_exps: {probe_state_path.name}")
        else:
            logger.log(f"  - [SKIP] Probe already trained: {probe_state_path.name}")
        return
    
    logger.log("  - Training new probe …")

    # Prepare dataset
    logger.log(f"  [DEBUG] Starting dataset preparation...")
    if job.rebuild_config is not None:
        logger.log(f"  [DEBUG] Using rebuild_config: {job.rebuild_config}")
        orig_ds = Dataset(job.train_dataset, model_name=config['model_name'], device=config['device'], seed=job.seed)
        logger.log(f"  [DEBUG] Created original dataset")
        
        # Filter data for off-policy experiments if model_check was run
        if not job.on_policy and 'model_check' in config:
            run_name = config.get('run_name', 'default_run')
            for check in config['model_check']:
                check_on = check['check_on']
                if isinstance(check_on, list):
                    datasets_to_check = check_on
                else:
                    datasets_to_check = [check_on] if check_on else []
                
                if job.train_dataset in datasets_to_check:
                    logger.log(f"  [DEBUG] Filtering data for off-policy experiment...")
                    orig_ds.filter_data_by_model_check(run_name, check['name'])
                    break
        
        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in job.rebuild_config and job.rebuild_config['llm_upsampling']:
            logger.log(f"  [DEBUG] Using LLM upsampling method")
            n_real_neg = job.rebuild_config.get('n_real_neg')
            n_real_pos = job.rebuild_config.get('n_real_pos')
            upsampling_factor = job.rebuild_config.get('upsampling_factor')
            
            run_name = str(results_dir).split('/')[-3]
            llm_csv_base_path = job.rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError("For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified")
            
            logger.log(f"  [DEBUG] Building LLM upsampled dataset...")
            train_ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=job.seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=job.val_size,
                test_size=job.test_size,
                llm_csv_base_path=llm_csv_base_path,
                only_test=False,
            )
            logger.log(f"  [DEBUG] Built LLM upsampled dataset")
        else:
            # Original rebuild_config logic
            logger.log(f"  [DEBUG] Using original rebuild_config logic")
            train_class_counts = job.rebuild_config.get('class_counts')
            train_class_percents = job.rebuild_config.get('class_percents')
            train_total_samples = job.rebuild_config.get('total_samples')
            logger.log(f"  [DEBUG] Building imbalanced train balanced eval dataset...")
            train_ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=job.val_size,
                test_size=job.test_size,
                seed=job.seed,  # Use the global seed passed to this function
                only_test=False,
            )
            logger.log(f"  [DEBUG] Built imbalanced train balanced eval dataset")
    else:
        logger.log(f"  [DEBUG] Using simple dataset creation")
        train_ds = Dataset(job.train_dataset, model_name=config['model_name'], device=config['device'], seed=job.seed)
        logger.log(f"  [DEBUG] Created dataset")
        
        # Filter data for off-policy experiments if model_check was run
        if not job.on_policy and 'model_check' in config:
            run_name = config.get('run_name', 'default_run')
            for check in config['model_check']:
                check_on = check['check_on']
                if isinstance(check_on, list):
                    datasets_to_check = check_on
                else:
                    datasets_to_check = [check_on] if check_on else []
                
                if job.train_dataset in datasets_to_check:
                    logger.log(f"  [DEBUG] Filtering data for off-policy experiment...")
                    train_ds.filter_data_by_model_check(run_name, check['name'])
                    break
        
        logger.log(f"  [DEBUG] Now splitting data...")
        train_ds.split_data(train_size=job.train_size, val_size=job.val_size, test_size=job.test_size, seed=job.seed)
        logger.log(f"  [DEBUG] Split data complete")

    # Get activations
    activation_type = get_activation_type_from_config(config_name)
    logger.log(f"  [DEBUG] Using activation_type: {activation_type}")
    
    # Get activation extraction format from config
    format_type = "r-no-it"
    if hasattr(job, 'format_type'):
        format_type = job.format_type
    elif hasattr(job, 'config') and 'activation_extraction' in job.config:
        format_type = job.config['activation_extraction'].get('format_type', 'r-no-it')
    
    # Get activations using the unified method
    train_result = train_ds.get_train_set_activations(job.layer, job.component, activation_type=activation_type, on_policy=job.on_policy, format_type=format_type)
    
    # Handle different return formats (with/without masks)
    if len(train_result) == 3:
        train_acts, train_masks, y_train = train_result
    else:
        train_acts, y_train = train_result
        train_masks = None
    
    logger.log(f"  [DEBUG] Got activations with type: {activation_type}")

    print(f"Train activations: {len(train_acts) if isinstance(train_acts, list) else train_acts.shape}")

    # Create probe based on architecture_name
    logger.log(f"  [DEBUG] Creating probe for architecture: {job.architecture_name}")
    
    # Convert probe config to dict
    probe_config_dict = asdict(job.probe_config)
    
    if job.architecture_name == "sklearn_linear":
        logger.log(f"  [DEBUG] Creating sklearn linear probe")
        # Sklearn linear probe
        probe = get_probe_architecture("sklearn_linear", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
        logger.log(f"  [DEBUG] Created sklearn linear probe")
        
        if retrain_with_best_hparams:
            logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
            # Load best hyperparameters from the dedicated best_hparams.json file
            if job.rebuild_config is not None:
                suffix = rebuild_suffix(job.rebuild_config)
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
            logger.log(f"  [DEBUG] Hyperparameter tuning not implemented in refactored version")
            # TODO: Implement hyperparameter tuning without find_best_fit
            best_params = {}
            logger.log(f"  [DEBUG] Skipped hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting probe normally...")
            # Pass masks only if they exist (for pre-aggregated activations, masks=None)
            if train_masks is not None:
                probe.fit(train_acts, y_train, train_masks)
            else:
                probe.fit(train_acts, y_train)
            logger.log(f"  [DEBUG] Fitted probe normally")
            
    elif job.architecture_name == "linear":
        logger.log(f"  [DEBUG] Creating PyTorch linear probe")
        # PyTorch linear probe
        probe = get_probe_architecture("linear", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
        logger.log(f"  [DEBUG] Created PyTorch linear probe")
        
        if retrain_with_best_hparams:
            logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
            if job.rebuild_config is not None:
                suffix = rebuild_suffix(job.rebuild_config)
                best_hparams_path = probe_save_dir / f"{probe_filename_base}_{suffix}_best_hparams.json"
            else:
                best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            if not best_hparams_path.exists():
                raise FileNotFoundError(f"Best hyperparameters file not found: {best_hparams_path}")
            with open(best_hparams_path, 'r') as f:
                best_params = json.load(f)
            logger.log(f"  [DEBUG] Fitting PyTorch linear probe with best hyperparameters...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted PyTorch linear probe with best hyperparameters")
        elif hyperparameter_tuning:
            logger.log(f"  [DEBUG] Hyperparameter tuning not implemented in refactored version")
            # TODO: Implement hyperparameter tuning without find_best_fit
            best_params = {}
            logger.log(f"  [DEBUG] Skipped hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting PyTorch linear probe normally...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted PyTorch linear probe normally")
            
    elif job.architecture_name == "attention":
        logger.log(f"  [DEBUG] Creating attention probe")
        # Attention probe
        probe = get_probe_architecture("attention", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
        logger.log(f"  [DEBUG] Created attention probe")
        
        if retrain_with_best_hparams:
            logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
            if job.rebuild_config is not None:
                suffix = rebuild_suffix(job.rebuild_config)
                best_hparams_path = probe_save_dir / f"{probe_filename_base}_{suffix}_best_hparams.json"
            else:
                best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            if not best_hparams_path.exists():
                raise FileNotFoundError(f"Best hyperparameters file not found: {best_hparams_path}")
            with open(best_hparams_path, 'r') as f:
                best_params = json.load(f)
            logger.log(f"  [DEBUG] Fitting attention probe with best hyperparameters...")
            probe.fit(train_acts, y_train)
            logger.log(f"  [DEBUG] Fitted attention probe with best hyperparameters")
        elif hyperparameter_tuning:
            logger.log(f"  [DEBUG] Hyperparameter tuning not implemented in refactored version")
            # TODO: Implement hyperparameter tuning without find_best_fit
            best_params = {}
            logger.log(f"  [DEBUG] Skipped hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting attention probe normally...")
            probe.fit(train_acts, y_train)
            logger.log(f"  [DEBUG] Fitted attention probe normally")
            
    elif job.architecture_name == "sae":
        logger.log(f"  [DEBUG] Creating SAE probe")
        # SAE probe
        probe = get_probe_architecture("sae", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
        logger.log(f"  [DEBUG] Created SAE probe")
        
        if retrain_with_best_hparams:
            logger.log(f"  - Using best hyperparameters from best_hparams.json for {config_name}.")
            if job.rebuild_config is not None:
                suffix = rebuild_suffix(job.rebuild_config)
                best_hparams_path = probe_save_dir / f"{probe_filename_base}_{suffix}_best_hparams.json"
            else:
                best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            if not best_hparams_path.exists():
                raise FileNotFoundError(f"Best hyperparameters file not found: {best_hparams_path}")
            with open(best_hparams_path, 'r') as f:
                best_params = json.load(f)
            logger.log(f"  [DEBUG] Fitting SAE probe with best hyperparameters...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted SAE probe with best hyperparameters")
        elif hyperparameter_tuning:
            logger.log(f"  [DEBUG] Hyperparameter tuning not implemented in refactored version")
            # TODO: Implement hyperparameter tuning without find_best_fit
            best_params = {}
            logger.log(f"  [DEBUG] Skipped hyperparameter tuning")
        else:
            logger.log(f"  [DEBUG] Fitting SAE probe normally...")
            probe.fit(train_acts, y_train, train_masks)
            logger.log(f"  [DEBUG] Fitted SAE probe normally")
            
    elif job.architecture_name == "mass_mean":
        logger.log(f"  [DEBUG] Creating mass mean probe")
        # Mass mean probe (non-trainable)
        probe = get_probe_architecture("mass_mean", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
        logger.log(f"  [DEBUG] Created mass mean probe")
        logger.log(f"  [DEBUG] Fitting mass mean probe...")
        probe.fit(train_acts, y_train)
        logger.log(f"  [DEBUG] Fitted mass mean probe")
        
    elif job.architecture_name == "act_sim":
        logger.log(f"  [DEBUG] Creating activation similarity probe")
        # Activation similarity probe (non-trainable)
        probe = get_probe_architecture("act_sim", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
        logger.log(f"  [DEBUG] Created activation similarity probe")
        logger.log(f"  [DEBUG] Fitting activation similarity probe...")
        probe.fit(train_acts, y_train)
        logger.log(f"  [DEBUG] Fitted activation similarity probe")
        
    else:
        raise ValueError(f"Unknown architecture_name: {job.architecture_name}")

    # Save probe
    logger.log(f"  [DEBUG] Saving probe state...")
    probe_save_dir.mkdir(parents=True, exist_ok=True)
    probe.save_state(probe_state_path)
    logger.log(f"  [DEBUG] Saved probe state")
    
    # Save metadata/config
    logger.log(f"  [DEBUG] Saving metadata...")
    meta = {
        'train_dataset_name': job.train_dataset,
        'layer': job.layer,
        'component': job.component,
        'architecture_name': job.architecture_name,
        'aggregation': extract_aggregation_from_config(config_name, job.architecture_name) if hasattr(probe, 'aggregation') else None,
        'config_name': config_name,
        'rebuild_config': copy.deepcopy(job.rebuild_config),
        'probe_state_path': str(probe_state_path),
        'on_policy': job.on_policy,
    }
    if hyperparameter_tuning:
        meta['hyperparameters'] = best_params
    with open(probe_json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.log(f"  [DEBUG] Saved metadata")
    logger.log(f"  - 🔥 Probe state saved to {probe_state_path.name}")
    logger.log(f"  [DEBUG] train_single_probe function completed successfully")

def evaluate_single_probe(job: ProbeJob, eval_dataset: str, config: Dict, results_dir: Path, cache_dir: Path, logger: Logger, only_test: bool, reevaluate: bool):
    """
    Evaluate a single probe on a given dataset.
    """
    # Create config name from architecture and probe config
    config_name = f"{job.architecture_name}_{job.probe_config.aggregation}" if hasattr(job.probe_config, 'aggregation') else job.architecture_name
    
    # Extract aggregation from config for results, to be backward compatible
    agg_name = extract_aggregation_from_config(config_name, job.architecture_name)

    probe_filename_base = get_probe_filename_prefix(job.train_dataset, job.architecture_name, job.layer, job.component, config_name)
    # Add hyperparameters to filename if they differ from defaults
    probe_filename_with_hparams = add_hyperparams_to_filename(probe_filename_base, job.probe_config)
    
    # The results_dir is already the specific evaluation directory (val_eval, test_eval, or gen_eval)
    # We need to look for probe files in the trained directory
    trained_dir = results_dir.parent / "trained"
    
    if job.rebuild_config is not None:
        suffix = rebuild_suffix(job.rebuild_config)
        probe_state_path = trained_dir / f"{probe_filename_with_hparams}_{suffix}_state.npz"
        eval_results_path = results_dir / f"eval_on_{eval_dataset}__{probe_filename_with_hparams}_{suffix}_seed{job.seed}_{agg_name}_results.json"
    else:
        probe_state_path = trained_dir / f"{probe_filename_with_hparams}_state.npz"
        eval_results_path = results_dir / f"eval_on_{eval_dataset}__{probe_filename_with_hparams}_seed{job.seed}_{agg_name}_results.json"

    if config.get('cache_activations', True) and eval_results_path.exists() and not reevaluate:
        logger.log("  - 😋 Using cached evaluation result ")
        with open(eval_results_path, "r") as f:
            return json.load(f)["metrics"]

    # Load probe
    probe_config_dict = asdict(job.probe_config)
    
    if job.architecture_name == "sklearn_linear":
        probe = get_probe_architecture("sklearn_linear", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
    elif job.architecture_name == "linear":
        probe = get_probe_architecture("linear", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
    elif job.architecture_name == "attention":
        probe = get_probe_architecture("attention", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
    elif job.architecture_name == "sae":
        probe = get_probe_architecture("sae", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
    elif job.architecture_name == "mass_mean":
        probe = get_probe_architecture("mass_mean", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
    elif job.architecture_name == "act_sim":
        probe = get_probe_architecture("act_sim", d_model=get_model_d_model(config['model_name']), device=config['device'], config=probe_config_dict)
    else:
        raise ValueError(f"Unknown architecture_name: {job.architecture_name}")
    
    probe.load_state(probe_state_path)
    
    # Get batch_size from probe config for evaluation
    # For SAE probes, use training_batch_size; for others, use batch_size
    if job.architecture_name == "sae":
        batch_size = probe_config_dict.get('training_batch_size', probe_config_dict.get('batch_size', 200))
    else:
        batch_size = probe_config_dict.get('batch_size', 200)

    # Prepare evaluation dataset
    if job.rebuild_config is not None:
        orig_ds = Dataset(eval_dataset, model_name=config['model_name'], device=config['device'], seed=job.seed, only_test=only_test)
        
        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in job.rebuild_config and job.rebuild_config['llm_upsampling']:
            n_real_neg = job.rebuild_config.get('n_real_neg')
            n_real_pos = job.rebuild_config.get('n_real_pos')
            upsampling_factor = job.rebuild_config.get('upsampling_factor')
            
            run_name = str(results_dir).split('/')[-3]
            llm_csv_base_path = job.rebuild_config.get('llm_csv_base_path', f'results/{run_name}')
            
            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError("For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified")
            
            eval_ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=job.seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=job.val_size,
                test_size=job.test_size,
                llm_csv_base_path=llm_csv_base_path,
                only_test=only_test,
            )
        else:
            # Original rebuild_config logic
            train_class_counts = job.rebuild_config.get('class_counts')
            train_class_percents = job.rebuild_config.get('class_percents')
            train_total_samples = job.rebuild_config.get('total_samples')
            eval_ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=job.val_size,
                test_size=job.test_size,
                seed=job.seed,
                only_test=only_test,
            )
    else:
        eval_ds = Dataset(eval_dataset, model_name=config['model_name'], device=config['device'], seed=job.seed, only_test=only_test)
        eval_ds.split_data(train_size=job.train_size, val_size=job.val_size, test_size=job.test_size, seed=job.seed)
    
    # Get activations
    activation_type = get_activation_type_from_config(config_name)
    logger.log(f"  [DEBUG] Using activation_type: {activation_type}")
    
    # Get activation extraction format from config
    format_type = "r-no-it"
    if hasattr(job, 'format_type'):
        format_type = job.format_type
    elif hasattr(job, 'config') and 'activation_extraction' in job.config:
        format_type = job.config['activation_extraction'].get('format_type', 'r-no-it')
    
    # Get activations using the unified method
    test_result = eval_ds.get_test_set_activations(job.layer, job.component, activation_type=activation_type, on_policy=job.on_policy, format_type=format_type)
    
    # Handle different return formats (with/without masks)
    if len(test_result) == 3:
        test_acts, test_masks, y_test = test_result
    else:
        test_acts, y_test = test_result
        test_masks = None
    
    print(f"Test activations: {len(test_acts) if isinstance(test_acts, list) else test_acts.shape}")
    
    # Calculate metrics based on score options
    combined_metrics = {}
    
    if 'all' in job.score_options:
        logger.log(f"  - 🤗 Calculating all examples metrics...")
        if job.architecture_name == "attention" or job.architecture_name == "act_sim":
            # Attention probes and activation similarity probes don't use masks
            all_metrics = probe.score(test_acts, y_test)
        else:
            # Other probes use masks
            all_metrics = probe.score(test_acts, y_test, masks=test_masks)
        combined_metrics["all_examples"] = all_metrics

    if 'filtered' in job.score_options:
        logger.log(f"  - 🤗 Calculating filtered metrics...")
        try:
            if job.architecture_name == "attention" or job.architecture_name == "act_sim":
                # Attention probes and activation similarity probes don't use masks
                filtered_metrics = probe.score_filtered(test_acts, y_test, dataset_name=eval_dataset, 
                                                      results_dir=results_dir, seed=job.seed, test_size=job.test_size)
            else:
                # Other probes use masks
                filtered_metrics = probe.score_filtered(test_acts, y_test, masks=test_masks, 
                                                      dataset_name=eval_dataset, results_dir=results_dir, 
                                                      seed=job.seed, test_size=job.test_size)
            combined_metrics["filtered_examples"] = filtered_metrics
        except Exception as e:
            logger.log(f"  - 😵‍💫Filtered scoring failed with error: {str(e)}")
            logger.log(f"  - 🔍 Debug info: eval_dataset_name={eval_dataset}, results_dir={results_dir}")
            # Set a default filtered metrics structure to indicate failure
            combined_metrics["filtered_examples"] = {
                "filtered": False,
                "error": str(e),
                "fallback_to_all": True
            }

    # Save metrics and per-datapoint scores/labels
    # Compute per-datapoint probe scores (logits) and labels
    if job.architecture_name == "attention" or job.architecture_name == "act_sim":
        # Attention probes and activation similarity probes don't use masks
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
    if 'filtered' in job.score_options and 'filtered_examples' in combined_metrics:
        filtered_metrics = combined_metrics["filtered_examples"]
        if filtered_metrics.get("filtered", False) and filtered_metrics.get("removed_count", 0) > 0:
            # Get filtered indices from the CSV file
            # Runthrough directory is in the run_name directory (results/{run_name}/runthrough_{dataset_name}/)
            # results_dir is like: results/{run_name}/seed_{seed}/...
            # So we need to go up to the run_name level
            run_name_dir = results_dir.parent.parent  # Go up two levels to get to run_name
            runthrough_dir = run_name_dir / f"runthrough_{eval_dataset}"
            
            # Check if runthrough directory exists
            if not runthrough_dir.exists():
                logger.log(f"  - 🚨 Runthrough directory not found: {runthrough_dir}")
                return combined_metrics
            
            # Find CSV files
            csv_files = list(runthrough_dir.glob("*logit_diff*.csv"))
            if not csv_files:
                logger.log(f"  - 🚨 No logit_diff CSV files found in: {runthrough_dir}")
                return combined_metrics
            
            logger.log(f"  - ✅ Found CSV file: {csv_files[0]}")
            
            # Read CSV and check for required column
            df = pd.read_csv(csv_files[0])
            if 'use_in_filtered_scoring' not in df.columns:
                logger.log(f"  - 🚨 Column 'use_in_filtered_scoring' not found in CSV")
                return combined_metrics
            
            # Align prompts between model check CSV and current test set
            # Model check runs on full dataset, but we need to match with current test set
            if 'prompt' not in df.columns:
                logger.log(f"  - 🚨 Column 'prompt' not found in CSV for prompt alignment")
                return combined_metrics
            
            # Create a mapping from prompt to use_in_filtered_scoring
            prompt_to_filter = dict(zip(df['prompt'], df['use_in_filtered_scoring']))
            
            # Get the current test prompts and align them
            test_prompts = [str(prompt) for prompt in eval_ds.X_test]  # Convert to strings for matching
            
            # Find which test examples should be included in filtered scoring
            filtered_indices = []
            matched_count = 0
            for i, test_prompt in enumerate(test_prompts):
                if test_prompt in prompt_to_filter:
                    matched_count += 1
                    if prompt_to_filter[test_prompt] == 1:
                        filtered_indices.append(i)
            
            filtered_indices = np.array(filtered_indices)
            
            # Log alignment statistics
            logger.log(f"  - 📊 Prompt alignment: {matched_count}/{len(test_prompts)} test prompts found in model check CSV")
            logger.log(f"  - 📊 Model check CSV has {len(df)} total prompts")
            
            if len(filtered_indices) == 0:
                logger.log(f"  - ⚠️ No examples passed the filter (all use_in_filtered_scoring values are 0)")
            else:
                logger.log(f"  - ✅ Successfully filtered {len(filtered_indices)} examples from {len(test_scores)} total")
                output_dict["filtered_scores"] = {
                    "scores": [test_scores[i] for i in filtered_indices],
                    "labels": [test_labels[i] for i in filtered_indices],
                    "filtered": True,
                    "filter_method": "use_in_filtered_scoring",
                    "original_size": len(test_scores),
                    "filtered_size": len(filtered_indices),
                    "removed_count": len(test_scores) - len(filtered_indices)
                }
    
    # Ensure the directory exists
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(eval_results_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    # Log the results
    if 'all' in job.score_options:
        all_metrics = combined_metrics.get("all_examples", combined_metrics)
        logger.log(f"  - ❤️‍🔥 Success! All examples metrics: {all_metrics}")
    
    if 'filtered' in job.score_options:
        filtered_metrics = combined_metrics.get("filtered_examples", {})
        if filtered_metrics.get("filtered", False):
            logger.log(f"  - 🙃 Filtered examples metrics: {filtered_metrics}")
        else:
            error_msg = filtered_metrics.get("error", "Unknown error")
            logger.log(f"  - 😵‍💫Filtered scoring failed: {error_msg}")
            logger.log(f"  - 📊 Falling back to using all examples for evaluation")
    
    return combined_metrics
