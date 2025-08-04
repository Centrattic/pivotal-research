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
from src.utils import should_skip_dataset, resample_params_to_str, get_dataset, get_effective_seeds, get_effective_seed_for_rebuild_config, generate_llm_upsampling_configs
from src.model_check.main import run_model_check
from transformer_lens import HookedTransformer

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

    # Get log file path from config, default to results_dir / "output.log"
    log_file_path = config.get('log_file', results_dir / "output.log")
    # If log_file is a string, convert to Path and make it relative to results_dir if it's not absolute
    if isinstance(log_file_path, str):
        log_file_path = Path(log_file_path)
        if not log_file_path.is_absolute():
            log_file_path = results_dir / log_file_path
    
    # Ensure the log file directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(log_file_path)

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

        # Step 1: Pre-flight check and gather all jobs (training + evaluation)
        logger.log("\n Performing Pre-flight Checks and Gathering Jobs")
        all_jobs = []
        all_dataset_names_to_check = set()

        for experiment in config['experiments']:
            experiment_name = experiment['name']
            train_sets = [experiment['train_on']]
            eval_sets = experiment['evaluate_on']
            
            all_dataset_names_to_check.update(d for d in train_sets if d in available_datasets)
            all_dataset_names_to_check.update(d for d in eval_sets if d in available_datasets)
            
            for train_dataset in train_sets:
                for arch_config in config.get('architectures', []):
                    architecture_name = arch_config['name']
                    for layer in config['layers']:
                        for component in config['components']:
                            for seed in all_seeds:
                                config_name = arch_config.get('config_name')
                                # Create a job tuple that includes all necessary info for both training and evaluation
                                job = {
                                    'experiment_name': experiment_name,
                                    'train_dataset': train_dataset,
                                    'layer': layer,
                                    'component': component,
                                    'architecture_name': architecture_name,
                                    'config_name': config_name,
                                    'seed': seed,
                                    'experiment': experiment,
                                    'arch_config': arch_config,
                                    'eval_datasets': eval_sets
                                }
                                all_jobs.append(job)

        # Step 2: Combined Training and Evaluation Phase
        logger.log("\n" + "="*25 + " TRAINING & EVALUATION PHASE " + "="*25)
        
        for i, job in enumerate(all_jobs):
            logger.log("-" * 80)
            logger.log(f"🫠 Job {i+1}/{len(all_jobs)}: {job['experiment_name']}, {job['train_dataset']}, {job['architecture_name']}, L{job['layer']}, {job['component']}, seed={job['seed']}")
            
            # Validate train dataset for this specific seed
            try:
                logger.log(f"Validating train dataset '{job['train_dataset']}' with seed {job['seed']}")
                train_data = get_dataset(job['train_dataset'], model, device, job['seed'])
                train_data.split_data(test_size=0.15, seed=job['seed'])
                if should_skip_dataset(job['train_dataset'], train_data, logger):
                    logger.log(f"  - Skipping job due to train dataset validation failure")
                    continue
            except Exception as e:
                logger.log(f"  - 💀 ERROR validating train dataset '{job['train_dataset']}' with seed {job['seed']}: {e}")
                continue
            
            # Create seed-specific directory structure
            seed_dir = results_dir / f"seed_{job['seed']}"
            experiment_dir = seed_dir / job['experiment_name']
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            metric = job['experiment'].get('metric', 'acc')
            score_options = job['experiment'].get('score', ['all'])
            
            # Handle rebuild configs
            rebuild_configs = []
            if job['experiment'] and 'rebuild_config' in job['experiment']:
                rc = job['experiment']['rebuild_config']
                if isinstance(rc, dict):
                    for group in rc.values():
                        rebuild_configs.extend(group)
                else:
                    rebuild_configs.extend(rc if isinstance(rc, list) else [rc])
            else:
                rebuild_configs = [None]
            
            # Handle contrast function
            contrast_fn = None
            if job['experiment'] and 'contrast_fn' in job['experiment']:
                contrast_fn = Dataset.load_contrast_fn(job['experiment']['contrast_fn'])
            
            # Process each rebuild config
            for rebuild_params in rebuild_configs:
                # Get effective seed for this rebuild_config
                effective_seed = get_effective_seed_for_rebuild_config(job['seed'], rebuild_params)
                
                # Check if this is a non-trainable probe
                if (job['architecture_name'].startswith('act_sim') or 
                    job['architecture_name'] in ['mass_mean', 'mass_mean_iid']):
                    
                    logger.log(f"  😋 Running non-trainable probe: {job['architecture_name']}")
                    
                    # Run non-trainable probe
                    run_non_trainable_probe(
                        model=model, d_model=d_model, train_dataset_name=job['train_dataset'],
                        layer=job['layer'], component=job['component'], 
                        architecture_name=job['architecture_name'], 
                        config_name=job['config_name'],
                        device=config['device'], use_cache=config['cache_activations'], 
                        seed=effective_seed,
                        results_dir=experiment_dir, cache_dir=cache_dir, logger=logger, 
                        retrain=retrain,
                        rebuild_config=rebuild_params, metric=metric, contrast_fn=contrast_fn
                    )
                else:
                    logger.log(f"  💅 Training probe: {job['architecture_name']}")
                    
                    # Train the probe
                    train_probe(
                        model=model, d_model=d_model, train_dataset_name=job['train_dataset'],
                        layer=job['layer'], component=job['component'], 
                        architecture_name=job['architecture_name'], 
                        config_name=job['config_name'],
                        device=config['device'], use_cache=config['cache_activations'], 
                        seed=effective_seed,
                        results_dir=experiment_dir, cache_dir=cache_dir, logger=logger, 
                        retrain=retrain,
                        hyperparameter_tuning=hyperparameter_tuning, 
                        rebuild_config=rebuild_params, metric=metric,
                        retrain_with_best_hparams=retrain_with_best_hparams,
                        contrast_fn=contrast_fn
                    )
                
                # Immediately evaluate the probe on all evaluation datasets
                logger.log(f"  🤔 Evaluating {job['config_name']} probe on {len(job['eval_datasets'])} datasets...")
                
                all_eval_results = {}
                for eval_dataset in job['eval_datasets']:
                    try:
                        # Validate eval dataset for this specific seed
                        logger.log(f"    Validating eval dataset '{eval_dataset}' with seed {job['seed']}")
                        eval_data = get_dataset(eval_dataset, model, device, job['seed'])
                        eval_data.split_data(test_size=0.15, seed=job['seed'])
                        if should_skip_dataset(eval_dataset, eval_data, logger):
                            logger.log(f"      - Skipping evaluation due to eval dataset validation failure")
                            continue
                        
                        # Check task compatibility (binary classification only)
                        train_n_classes = getattr(train_data, 'n_classes', None)
                        eval_n_classes = getattr(eval_data, 'n_classes', None)
                        if train_n_classes != eval_n_classes:
                            logger.log(f"      - 🫡 Skipping evaluation of probe from '{job['train_dataset']}' on '{eval_dataset}' due to class count mismatch.")
                            continue
                            
                    except Exception as e:
                        logger.log(f"      - 💀 ERROR validating eval dataset '{eval_dataset}' with seed {job['seed']}: {e}")
                        continue
                    
                    # Check if probe directory exists
                    if rebuild_params is not None:
                        probe_save_dir = experiment_dir / f"dataclass_exps_{job['train_dataset']}"
                    else:
                        probe_save_dir = experiment_dir / f"train_{job['train_dataset']}"
                    
                    if not probe_save_dir.exists():
                        logger.log(f"      - [SKIP] Probe dir does not exist: {probe_save_dir}")
                        continue
                    
                    # Evaluate the probe
                    try:
                        metrics = evaluate_probe(
                            train_dataset_name=job['train_dataset'], 
                            eval_dataset_name=eval_dataset,
                            layer=job['layer'], component=job['component'], 
                            architecture_config=job['arch_config'],
                            results_dir=experiment_dir, logger=logger, seed=effective_seed,
                            model=model, d_model=d_model, device=config['device'],
                            use_cache=config['cache_activations'], cache_dir=cache_dir, 
                            reevaluate=reevaluate,
                            score_options=score_options, rebuild_config=rebuild_params, 
                            return_metrics=True,
                            contrast_fn=contrast_fn
                        )
                        
                        key = resample_params_to_str(rebuild_params)
                        if key not in all_eval_results:
                            all_eval_results[key] = {}
                        all_eval_results[key][eval_dataset] = metrics
                        
                        logger.log(f"      🤩 Evaluation complete for {eval_dataset}")
                        
                    except Exception as e:
                        logger.log(f"      - 💀 ERROR evaluating probe on '{eval_dataset}': {e}")
                        continue
                
                # Save evaluation results for this rebuild config
                if all_eval_results:
                    probe_filename_base = get_probe_filename_prefix(
                        job['train_dataset'], job['architecture_name'], 
                        job['layer'], job['component'], job['config_name'], contrast_fn
                    )
                    
                    # Determine the correct directory to save results
                    if rebuild_params is not None:
                        eval_results_path = experiment_dir / f"dataclass_exps_{job['train_dataset']}" / f"eval_on_all__{probe_filename_base}_allres.json"
                        eval_results_path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        eval_results_path = experiment_dir / f"train_{job['train_dataset']}" / f"eval_on_all__{probe_filename_base}_allres.json"
                        eval_results_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(eval_results_path, "w") as f:
                        json.dump(all_eval_results, f, indent=2)
                    
                    logger.log(f"  🤗 Saved evaluation results to {eval_results_path}")
                
                logger.log(f"  😜 Completed job {i+1}/{len(all_jobs)}")
                
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        logger.log("=" * 80)
        logger.log("\U0001f979 Run finished. Closing log file.")
        logger.close()

if __name__ == "__main__":
    main()
