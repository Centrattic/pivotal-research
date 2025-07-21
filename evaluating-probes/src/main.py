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

args = parser.parse_args()
global config_yaml, retrain, reevaluate, hyperparameter_tuning
config_yaml = args.config + "_config.yaml"
retrain = args.t
reevaluate = args.e
hyperparameter_tuning = args.ht

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
from src.runner import train_probe, evaluate_probe
from src.data import Dataset, get_available_datasets, load_combined_classification_datasets
from src.logger import Logger
from src.utils import should_skip_dataset
from src.model_check.main import run_model_check
from transformer_lens import HookedTransformer

# To force code to run on cuda:1, if exists
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_dataset(name, model, device, seed):
    # if name == "single_all":
    #     return load_combined_classification_datasets(seed)
    # else:
    return Dataset(name, model=model, device=device, seed=seed)

def resample_params_to_str(params):
    if params is None:
        return "original"
    if 'class_counts' in params:
        cc = params['class_counts']
        cc_str = '_'.join([f"class{cls}_{cc[cls]}" for cls in sorted(cc)])
        return f"{cc_str}_seed{params.get('seed', 42)}"
    elif 'class_percents' in params:
        cp = params['class_percents']
        cp_str = '_'.join([f"class{cls}_{int(cp[cls]*100)}pct" for cls in sorted(cp)])
        return f"{cp_str}_total{params['total_samples']}_seed{params.get('seed', 42)}"
    else:
        return f"custom_seed{params.get('seed', 42)}"

def main():
    # Clear GPU memory and set device
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    
    global config_yaml, retrain, reevaluate, hyperparameter_tuning

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
        model = HookedTransformer.from_pretrained(config['model_name'], device)
        d_model = model.cfg.d_model

        global_seed = int(config.get('seed', 42))
        available_datasets = get_available_datasets()

        # Step 1: Pre-flight check and gather all unique training jobs
        logger.log("\n Performing Pre-flight Checks and Gathering Jobs")
        training_jobs = set()
        all_dataset_names_to_check = set()

        for experiment in config['experiments']:
            train_sets = [experiment['train_on']]
            if experiment['train_on'] == "all":
                train_sets = available_datasets
            all_dataset_names_to_check.update(train_sets)

            eval_sets = experiment['evaluate_on']
            if "all" in eval_sets: eval_sets = available_datasets
            all_dataset_names_to_check.update(d for d in eval_sets if d != 'self')

            for train_dataset in train_sets:
                for arch_config in config.get('architectures', []):
                    for layer in config['layers']:
                        for component in config['components']:
                            # Determine config_name based on aggregation for linear probes
                            if arch_config['name'] == 'linear':
                                agg = arch_config['aggregation']
                                config_name = arch_config.get('config_name') or f"linear_{agg}"
                            else:
                                config_name = arch_config.get('config_name')
                            training_jobs.add((train_dataset, layer, component, arch_config['name'], arch_config['aggregation'], config_name))

        # Check all datasets that will be used for either training or evaluation
        valid_dataset_metadata = {}
        for dataset_name in sorted(list(all_dataset_names_to_check)):
            try:
                logger.log(dataset_name)
                data = get_dataset(dataset_name, model, device, global_seed)
                data.split_data(test_size=0.15, seed=global_seed)
                # logger.log("got here)")
                if should_skip_dataset(dataset_name, data, logger):
                    continue
                valid_dataset_metadata[dataset_name] = {
                    'task_type': getattr(data, 'task_type', None), 
                    'n_classes': getattr(data, 'n_classes', None)
                }
            except Exception as e:
                logger.log(f"  - 💀 ERROR checking dataset '{dataset_name}': {e}")

        valid_training_jobs = [job for job in training_jobs if job[0] in valid_dataset_metadata]

        # Step 2: Training Phase
        logger.log("\n" + "="*25 + " TRAINING PHASE " + "="*25)
        for i, (train_ds, layer, comp, arch_name, arch_agg, conf_name) in enumerate(valid_training_jobs):
            logger.log("-" * 60)
            logger.log(f"🫠 Training job {i+1}/{len(valid_training_jobs)}: {train_ds}, {arch_name}, L{layer}, {comp}")
            # Find the experiment for this job
            experiment = next((exp for exp in config['experiments'] if exp['train_on'] == train_ds), None)
            # Flatten grouped rebuild_config if present
            rebuild_configs = [None]
            if experiment and 'rebuild_config' in experiment:
                rc = experiment['rebuild_config']
                if isinstance(rc, dict):
                    for group in rc.values():
                        rebuild_configs.extend(group)
                else:
                    rebuild_configs.extend(rc if isinstance(rc, list) else [rc]) # Always include the None case
            for rebuild_params in rebuild_configs:
                train_probe(
                    model=model, d_model=d_model, train_dataset_name=train_ds,
                    layer=layer, component=comp, architecture_name=arch_name, config_name=conf_name,
                    device=config['device'], aggregation=arch_agg, use_cache=config['cache_activations'], seed=global_seed,
                    results_dir=results_dir, cache_dir=cache_dir, logger=logger, retrain=retrain,
                    hyperparameter_tuning=hyperparameter_tuning, rebuild_config=rebuild_params
                )

        # Step 3: Evaluation Phase
        logger.log("\n" + "="*25 + " EVALUATION PHASE " + "="*25)
        for experiment in config['experiments']:
            train_sets = [experiment['train_on']]
            if experiment['train_on'] == "all": train_sets = available_datasets
            score_options = experiment.get('score', ['all'])
            rebuild_configs = []
            if 'rebuild_config' in experiment:
                rc = experiment['rebuild_config']
                if isinstance(rc, dict):
                    for group in rc.values():
                        rebuild_configs.extend(group)
                else:
                    rebuild_configs = rc
            else:
                rebuild_configs = [None]
            for train_dataset in train_sets:
                if train_dataset not in valid_dataset_metadata: continue
                eval_sets = experiment['evaluate_on']
                if "all" in eval_sets: eval_sets = available_datasets
                if "self" in eval_sets: eval_sets = [d if d != "self" else train_dataset for d in eval_sets]
                for eval_dataset in eval_sets:
                    if eval_dataset not in valid_dataset_metadata: continue
                    train_meta = valid_dataset_metadata[train_dataset]
                    eval_meta = valid_dataset_metadata[eval_dataset]
                    if train_meta['task_type'] != eval_meta['task_type'] or train_meta['n_classes'] != eval_meta['n_classes']:
                        logger.log(f"  - 🫡  Skipping evaluation of probe from '{train_dataset}' on '{eval_dataset}' due to task mismatch.")
                        continue
                    for arch_config in config.get('architectures', []):
                        for layer in config['layers']:
                            for component in config['components']:
                                all_eval_results = {}
                                for rebuild_params in rebuild_configs:
                                    # Determine probe_save_dir based on whether this is a dataclass_exps probe
                                    if rebuild_params is not None:
                                        probe_save_dir = results_dir / f"dataclass_exps_{train_dataset}"
                                    else:
                                        probe_save_dir = results_dir / f"train_{train_dataset}"
                                    # Ensure probe_save_dir exists (skip if not trained)
                                    if not probe_save_dir.exists():
                                        logger.log(f"  - [SKIP] Probe dir does not exist: {probe_save_dir}")
                                        continue
                                    metrics = evaluate_probe(
                                        train_dataset_name=train_dataset, eval_dataset_name=eval_dataset,
                                        layer=layer, component=component, architecture_config=arch_config,
                                        aggregation=arch_config['aggregation'], results_dir=results_dir, logger=logger, seed=global_seed,
                                        model=model, d_model=d_model, device=config['device'],
                                        use_cache=config['cache_activations'], cache_dir=cache_dir, reevaluate=reevaluate,
                                        score_options=score_options, rebuild_config=rebuild_params, return_metrics=True
                                    )
                                    key = resample_params_to_str(rebuild_params)
                                    all_eval_results[key] = metrics
                                # Save all results for this probe/dataset/arch/layer/component combo
                                probe_filename_base = f"{train_dataset}_{arch_config['name']}_{arch_config['aggregation']}_L{layer}_{component}"
                                eval_results_path = results_dir / f"train_{train_dataset}" / f"eval_on_{eval_dataset}__{probe_filename_base}_allres.json"
                                with open(eval_results_path, "w") as f:
                                    json.dump(all_eval_results, f, indent=2)
                                # If this was a dataclass_exps probe, also save in that directory
                                if any(rebuild_configs):
                                    dataclass_eval_results_path = results_dir / f"dataclass_exps_{train_dataset}" / f"eval_on_{eval_dataset}__{probe_filename_base}_allres.json"
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
