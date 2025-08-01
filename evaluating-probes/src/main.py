import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import torch
import os
import sys
import json

# Set CUDA device BEFORE importing transformer_lens
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument('-e', action='store_true')
parser.add_argument('-t', action='store_true')
parser.add_argument('-ht', action='store_true', help='Enable hyperparameter tuning (Optuna)')
parser.add_argument('-bh', action='store_true', help='Retrain probes using best hyperparameters from previous tuning')

args = parser.parse_args()
global config_yaml, retrain, reevaluate, hyperparameter_tuning, retrain_with_best_hparams
config_yaml = args.config + "_config.yaml"
retrain = args.t
reevaluate = args.e
hyperparameter_tuning = args.ht
retrain_with_best_hparams = args.bh

# Load config early to set CUDA device before any CUDA operations
try:
    with open(f"configs/{config_yaml}", "r") as f:
        config = yaml.safe_load(f)
    device = config.get("device")
    if device and "cuda" in device:
        cuda_id = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
        # Update the device string to use cuda:0 since we're now only showing one device
        config["device"] = "cuda:0"
        print(f"Set CUDA_VISIBLE_DEVICES to {cuda_id}, updated device to cuda:0")
except Exception as e:
    print(f"Warning: Could not set CUDA device early: {e}")

# Now import transformer_lens after setting CUDA device
from src.runner import train_probe, evaluate_probe, get_probe_filename_prefix, run_non_trainable_probe
from src.data import Dataset, get_available_datasets
from src.logger import Logger
from src.utils import should_skip_dataset, resample_params_to_str
from src.model_check.main import run_model_check
from transformer_lens import HookedTransformer

def get_dataset(name, model, device, seed):
    return Dataset(name, model=model, device=device, seed=seed)

def get_effective_seeds(config):
    """Extract seeds from config, supporting both single seed and multiple seeds.
    Returns a list of seeds for uniform processing.
    """
    if 'seeds' in config:
        # Multiple seeds specified
        seeds = config['seeds']
        if isinstance(seeds, list):
            return seeds
        else:
            # Handle case where seeds might be a single value
            return [seeds]
    elif 'seed' in config:
        # Single seed specified (backward compatibility)
        return [config['seed']]
    else:
        # Default seed if none specified
        return [42]

def get_effective_seed_for_rebuild_config(global_seed, rebuild_config):
    """
    For LLM upsampling experiments, rebuild_config seed overrides global seed.
    For other experiments, use global seed.
    """
    if rebuild_config and 'llm_upsampling' in rebuild_config and rebuild_config['llm_upsampling']:
        # LLM upsampling: rebuild_config seed takes precedence
        return rebuild_config.get('seed', global_seed)
    else:
        # Regular experiments: global seed takes precedence
        return global_seed

def generate_llm_upsampling_configs(n_real_neg, n_real_pos_list, upsampling_factors, seed):
    """
    Generate rebuild_configs for LLM upsampling experiments.
    Creates a config for each combination of n_real_pos and upsampling_factor.
    
    Args:
        n_real_neg: Fixed number of real negative samples to use
        n_real_pos_list: List of real positive sample counts to try (e.g., [1, 2, 3, 4, 5])
        upsampling_factors: List of upsampling factors to try (e.g., [1, 2, 3, 4, 5])
        seed: Random seed for the experiments
    """
    configs = []
    for n_real_pos in n_real_pos_list:
        for factor in upsampling_factors:
            configs.append({
                'llm_upsampling': True,
                'n_real_neg': n_real_neg,
                'n_real_pos': n_real_pos,
                'upsampling_factor': factor,
                'seed': seed  # Override global seed for LLM experiments
            })
    return configs

def main():
    # Clear GPU memory and set device
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    
    global config_yaml, retrain, reevaluate, hyperparameter_tuning, retrain_with_best_hparams

    # Load config (already loaded at top, but reload to be safe)
    try:
        with open(f"configs/{config_yaml}", "r") as f:
            config = yaml.safe_load(f)
        # Re-apply device fix if needed
        # Make them all cuda:0 in config! Or just cuda
        device = config.get("device")
        if device and "cuda" in device and "1" in device:
            # If we're using cuda:1, make sure it's updated to cuda:0 after visible devices cut
            config["device"] = "cuda:0"
            print(f"Updated device from {device} to cuda:0")
    except: 
        print(f"A config of name {config_yaml} does not exist.")

    run_name = config.get('run_name', 'default_run')
    results_dir = Path("results") / run_name
    cache_dir = Path("activation_cache") / config['model_name']
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(results_dir / "output.log")

    # Get all seeds to process
    all_seeds = get_effective_seeds(config)
    logger.log(f"Processing {len(all_seeds)} seeds: {all_seeds}")

    # Run model check first if present
    if 'model_check' in config:
        run_name = config.get('run_name', 'default_run')
        all_checks_done = True
        for check in config['model_check']:
            ds_name = check['check_on']
            runthrough_dir = Path(f"results/{run_name}/runthrough_{ds_name}")
            if not runthrough_dir.exists():
                all_checks_done = False
                break
        if not all_checks_done:
            logger.log("\n=== Running model_check before main pipeline ===")
            run_model_check(config)
            logger.log("=== model_check complete ===\n")
        else:
            logger.log("\n=== Skipping model_check: all runthrough directories already exist ===\n")

    try:
        # Single model instance
        logger.log(f"Loading model '{config['model_name']}' to get config...")
        device = config.get("device")
        model = HookedTransformer.from_pretrained(config['model_name'], device) # Load to get activations if needed
        d_model = model.cfg.d_model

        available_datasets = get_available_datasets()

        # Step 1: Pre-flight check and gather all unique training jobs
        logger.log("\n Performing Pre-flight Checks and Gathering Jobs")
        training_jobs = set()
        all_dataset_names_to_check = set()

        for experiment in config['experiments']:
            experiment_name = experiment['name']
            experiment_dir = results_dir / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            train_sets = [experiment['train_on']]
            # if experiment['train_on'] == "all":
            #     train_sets = available_datasets
            all_dataset_names_to_check.update(d for d in train_sets if d in available_datasets)

            eval_sets = experiment['evaluate_on']
            # if "all" in eval_sets: eval_sets = available_datasets
            all_dataset_names_to_check.update(d for d in eval_sets if d in available_datasets) # if d != 'self')

            # logger.log(available_datasets)
            # logger.log(all_dataset_names_to_check)
            # logger.log(train_sets)
            # logger.log(eval_sets)
            
            for train_dataset in train_sets:
                for arch_config in config.get('architectures', []):
                    architecture_name = arch_config['name']
                    # Skip non-trainable architectures in the training phase
                    if architecture_name.startswith('act_sim') or architecture_name in ['mass_mean', 'mass_mean_iid']:
                        continue
                    
                    for layer in config['layers']:
                        for component in config['components']:
                            for seed in all_seeds:
                                config_name = arch_config.get('config_name')
                                training_jobs.add((experiment_name, train_dataset, layer, component, architecture_name, config_name, seed))

        # Step 2: Training Phase
        logger.log("\n" + "="*25 + " TRAINING PHASE " + "="*25)
        for i, (experiment_name, train_ds, layer, comp, arch_name, conf_name, seed) in enumerate(training_jobs):
            logger.log("-" * 60)
            logger.log(f"🫠 Training job {i+1}/{len(training_jobs)}: {experiment_name}, {train_ds}, {arch_name}, L{layer}, {comp}, seed={seed}")
            
            # Validate dataset for this specific seed
            try:
                logger.log(f"Validating dataset '{train_ds}' with seed {seed}")
                data = get_dataset(train_ds, model, device, seed)
                data.split_data(test_size=0.15, seed=seed) # necessary for checking
                if should_skip_dataset(train_ds, data, logger):
                    logger.log(f"  - Skipping training job due to dataset validation failure")
                    continue
            except Exception as e:
                logger.log(f"  - 💀 ERROR validating dataset '{train_ds}' with seed {seed}: {e}")
                continue
            
            experiment = next((exp for exp in config['experiments'] if exp['name'] == experiment_name), None)
            
            # Create seed-specific directory structure
            seed_dir = results_dir / f"seed_{seed}"
            experiment_dir = seed_dir / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            metric = experiment.get('metric', 'acc')
            rebuild_configs = []
            if experiment and 'rebuild_config' in experiment:
                # For experiments with rebuild_config, only train the rebuild_config variants
                # Do NOT include None (original training)
                rc = experiment['rebuild_config']
                if isinstance(rc, dict):
                    for group in rc.values():
                        rebuild_configs.extend(group)
                else:
                    rebuild_configs.extend(rc if isinstance(rc, list) else [rc])
            else:
                # For experiments without rebuild_config, train the original probe
                rebuild_configs = [None]
            for rebuild_params in rebuild_configs:
                contrast_fn = None
                if experiment and 'contrast_fn' in experiment:
                    contrast_fn = Dataset.load_contrast_fn(experiment['contrast_fn'])
                
                # Get effective seed for this rebuild_config
                effective_seed = get_effective_seed_for_rebuild_config(seed, rebuild_params)
                
                train_probe(
                    model=model, d_model=d_model, train_dataset_name=train_ds,
                    layer=layer, component=comp, architecture_name=arch_name, config_name=conf_name,
                    device=config['device'], use_cache=config['cache_activations'], seed=effective_seed,
                    results_dir=experiment_dir, cache_dir=cache_dir, logger=logger, retrain=retrain,
                    hyperparameter_tuning=hyperparameter_tuning, rebuild_config=rebuild_params, metric=metric,
                    retrain_with_best_hparams=retrain_with_best_hparams,
                    contrast_fn=contrast_fn
                )

        # Step 2.5: Non-Trainable Probes Phase
        logger.log("\n" + "="*25 + " NON-TRAINABLE PROBES PHASE " + "="*25)
        
        # Identify non-trainable probe architectures
        non_trainable_architectures = []
        for arch_config in config.get('architectures', []):
            arch_name = arch_config['name']
            if arch_name.startswith('act_sim') or arch_name in ['mass_mean', 'mass_mean_iid']:
                non_trainable_architectures.append(arch_config)
        
        if non_trainable_architectures:
            logger.log(f"Found {len(non_trainable_architectures)} non-trainable probe architectures: {[arch['name'] for arch in non_trainable_architectures]}")
            
            for experiment in config['experiments']:
                experiment_name = experiment['name']
                train_sets = [experiment['train_on']]
                all_dataset_names_to_check.update(d for d in train_sets if d in available_datasets)
                
                eval_sets = experiment['evaluate_on']
                all_dataset_names_to_check.update(d for d in eval_sets if d in available_datasets)
                
                for train_dataset in train_sets:
                    for arch_config in non_trainable_architectures:
                        for layer in config['layers']:
                            for component in config['components']:
                                for seed in all_seeds:
                                    # Validate dataset for this specific seed
                                    try:
                                        logger.log(f"Validating dataset '{train_dataset}' for non-trainable probe with seed {seed}")
                                        data = get_dataset(train_dataset, model, device, seed)
                                        data.split_data(test_size=0.15, seed=seed)
                                        if should_skip_dataset(train_dataset, data, logger):
                                            logger.log(f"  - Skipping non-trainable probe due to dataset validation failure")
                                            continue
                                    except Exception as e:
                                        logger.log(f"  - 💀 ERROR validating dataset '{train_dataset}' for non-trainable probe with seed {seed}: {e}")
                                        continue
                                    
                                    # Create seed-specific directory structure
                                    seed_dir = results_dir / f"seed_{seed}"
                                    experiment_dir = seed_dir / experiment_name
                                    experiment_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    metric = experiment.get('metric', 'acc')
                                    rebuild_configs = []
                                    if experiment and 'rebuild_config' in experiment:
                                        # For experiments with rebuild_config, only train the rebuild_config variants
                                        # Do NOT include None (original training)
                                        rc = experiment['rebuild_config']
                                        if isinstance(rc, dict):
                                            for group in rc.values():
                                                rebuild_configs.extend(group)
                                        else:
                                            rebuild_configs.extend(rc if isinstance(rc, list) else [rc])
                                    else:
                                        # For experiments without rebuild_config, train the original probe
                                        rebuild_configs = [None]
                                    
                                    for rebuild_params in rebuild_configs:
                                        contrast_fn = None
                                        if experiment and 'contrast_fn' in experiment:
                                            contrast_fn = Dataset.load_contrast_fn(experiment['contrast_fn'])
                                        
                                        # Get effective seed for this rebuild_config
                                        effective_seed = get_effective_seed_for_rebuild_config(seed, rebuild_params)
                                        
                                        # Run non-trainable probe
                                        run_non_trainable_probe(
                                            model=model, d_model=d_model, train_dataset_name=train_dataset,
                                            layer=layer, component=component, architecture_name=arch_config['name'], config_name=arch_config.get('config_name'),
                                            device=config['device'], use_cache=config['cache_activations'], seed=effective_seed,
                                            results_dir=experiment_dir, cache_dir=cache_dir, logger=logger, retrain=retrain,
                                            rebuild_config=rebuild_params, metric=metric, contrast_fn=contrast_fn
                                        )
        else:
            logger.log("No non-trainable probe architectures found.")

        # Step 3: Evaluation Phase
        logger.log("\n" + "="*25 + " EVALUATION PHASE " + "="*25)
        for experiment in config['experiments']:
            experiment_name = experiment['name']
            train_sets = [experiment['train_on']]
            # if experiment['train_on'] == "all": 
            #     train_sets = available_datasets
            score_options = experiment.get('score', ['all'])
            rebuild_configs = []
            if 'rebuild_config' in experiment:
                # For experiments with rebuild_config, only evaluate the rebuild_config variants
                # Do NOT include None (original training)
                rc = experiment['rebuild_config']
                if isinstance(rc, dict):
                    for group in rc.values():
                        rebuild_configs.extend(group)
                else:
                    rebuild_configs = rc
            else:
                # For experiments without rebuild_config, evaluate the original probe
                rebuild_configs = [None]
            for train_dataset in train_sets:
                eval_sets = experiment['evaluate_on']
                # if "all" in eval_sets: eval_sets = available_datasets
                # if "self" in eval_sets: eval_sets = [d if d != "self" else train_dataset for d in eval_sets]
                for eval_dataset in eval_sets:
                    for arch_config in config.get('architectures', []):
                        for layer in config['layers']:
                            for component in config['components']:
                                for seed in all_seeds:
                                    # Validate both train and eval datasets for this specific seed
                                    try:
                                        logger.log(f"Validating datasets for evaluation: train='{train_dataset}', eval='{eval_dataset}' with seed {seed}")
                                        train_data = get_dataset(train_dataset, model, device, seed)
                                        train_data.split_data(test_size=0.15, seed=seed)
                                        if should_skip_dataset(train_dataset, train_data, logger):
                                            logger.log(f"  - Skipping evaluation due to train dataset validation failure")
                                            continue
                                        
                                        eval_data = get_dataset(eval_dataset, model, device, seed)
                                        eval_data.split_data(test_size=0.15, seed=seed)
                                        if should_skip_dataset(eval_dataset, eval_data, logger):
                                            logger.log(f"  - Skipping evaluation due to eval dataset validation failure")
                                            continue
                                        
                                        # Check task compatibility
                                        train_meta = {
                                            'task_type': getattr(train_data, 'task_type', None), 
                                            'n_classes': getattr(train_data, 'n_classes', None)
                                        }
                                        eval_meta = {
                                            'task_type': getattr(eval_data, 'task_type', None), 
                                            'n_classes': getattr(eval_data, 'n_classes', None)
                                        }
                                        if train_meta['task_type'] != eval_meta['task_type'] or train_meta['n_classes'] != eval_meta['n_classes']:
                                            logger.log(f"  - 🫡  Skipping evaluation of probe from '{train_dataset}' on '{eval_dataset}' due to task mismatch.")
                                            continue
                                            
                                    except Exception as e:
                                        logger.log(f"  - 💀 ERROR validating datasets for evaluation with seed {seed}: {e}")
                                        continue
                                    
                                    # Create seed-specific directory structure
                                    seed_dir = results_dir / f"seed_{seed}"
                                    experiment_dir = seed_dir / experiment_name
                                    experiment_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    all_eval_results = {}
                                    for rebuild_params in rebuild_configs:
                                        if rebuild_params is not None:
                                            probe_save_dir = experiment_dir / f"dataclass_exps_{train_dataset}"
                                        else:
                                            probe_save_dir = experiment_dir / f"train_{train_dataset}"
                                        if not probe_save_dir.exists():
                                            logger.log(f"  - [SKIP] Probe dir does not exist: {probe_save_dir}")
                                            continue
                                        contrast_fn = None
                                        if 'contrast_fn' in experiment:
                                            contrast_fn = Dataset.load_contrast_fn(experiment['contrast_fn'])
                                        
                                        # Get effective seed for this rebuild_config
                                        effective_seed = get_effective_seed_for_rebuild_config(seed, rebuild_params)
                                        
                                        metrics = evaluate_probe(
                                            train_dataset_name=train_dataset, eval_dataset_name=eval_dataset,
                                            layer=layer, component=component, architecture_config=arch_config,
                                            results_dir=experiment_dir, logger=logger, seed=effective_seed,
                                            model=model, d_model=d_model, device=config['device'],
                                            use_cache=config['cache_activations'], cache_dir=cache_dir, reevaluate=reevaluate,
                                            score_options=score_options, rebuild_config=rebuild_params, return_metrics=True,
                                            contrast_fn=contrast_fn
                                        )
                                        key = resample_params_to_str(rebuild_params)
                                        all_eval_results[key] = metrics
                                    probe_filename_base = get_probe_filename_prefix(train_dataset, arch_config['name'], layer, component, arch_config.get('config_name'), contrast_fn)
                                    eval_results_path = experiment_dir / f"train_{train_dataset}" / f"eval_on_{eval_dataset}__{probe_filename_base}_allres.json"
                                    with open(eval_results_path, "w") as f:
                                        json.dump(all_eval_results, f, indent=2)
                                    if any(rebuild_configs):
                                        dataclass_eval_results_path = experiment_dir / f"dataclass_exps_{train_dataset}" / f"eval_on_{eval_dataset}__{probe_filename_base}_allres.json"
                                        dataclass_eval_results_path.parent.mkdir(parents=True, exist_ok=True)
                                        with open(dataclass_eval_results_path, "w") as f:
                                            json.dump(all_eval_results, f, indent=2)
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        logger.log("=" * 60)
        logger.log("\U0001f979 Run finished. Closing log file.")
        logger.close()

if __name__ == "__main__":
    main()
