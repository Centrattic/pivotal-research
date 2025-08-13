import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import json
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional
from joblib import Parallel, delayed
import sys
import os

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
from src.data import Dataset, get_available_datasets, get_model_d_model
from src.runner import evaluate_probe, train_probe, run_non_trainable_probe
from src.logger import Logger
from src.utils import should_skip_dataset, resample_params_to_str, get_effective_seeds, get_effective_seed_for_rebuild_config, generate_llm_upsampling_configs
from src.model_check.main import run_model_check
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def extract_activations_for_dataset_with_llm(model, tokenizer, model_name, dataset_name, layer, component, device, seed, logger, run_name, include_llm_samples=True):
    """
    Extract activations for a dataset using the full model.
    This is phase 1: ensure all activations are cached.
    Also pre-compute aggregated activations for faster loading later.
    If include_llm_samples=True, also extract activations for any LLM upsampled samples found.
    """
    logger.log(f"  - Extracting activations for {dataset_name}, L{layer}, {component}")
    
    try:
        # Create dataset with full model to extract activations
        ds = Dataset(dataset_name, model=model, tokenizer=tokenizer, model_name=model_name, device=device, seed=seed)
        
        # Ensure all dataset texts are cached (simpler, unified path)
        logger.log(f"    - Ensuring full activations cached for dataset texts...")
        dataset_texts = ds.X.tolist()
        newly_added = ds.act_manager.ensure_texts_cached(dataset_texts, layer, component)
        logger.log(f"    - Cached {newly_added} new activations (dataset)")

        # If including LLM samples, scan results/<run_name>/seed_*/llm_samples/samples_*.csv for prompts and cache them
        llm_samples_added = 0
        if include_llm_samples:
            try:
                results_dir = Path('results') / run_name
                llm_texts = []
                for seed_dir in results_dir.glob('seed_*'):
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

def create_readonly_dataset(model_name, dataset_name, device, seed):
    """
    Create a dataset with read-only activation manager.
    This is phase 2: use existing caches without loading the model.
    """
    return Dataset(dataset_name, model_name=model_name, device=device, seed=seed)

def process_dataset_job(dataset_job, config, retrain, hyperparameter_tuning, retrain_with_best_hparams, reevaluate):
    """
    Processes a single dataset job (train and evaluate all probes on one dataset).
    This function is designed to be called by joblib in parallel.
    Uses read-only activation managers since activations are already cached.
    """
    # Create a separate logger for this process
    log_file_path = dataset_job['log_file_path']
    logger = Logger(log_file_path)

    logger.log("-" * 80)
    rebuild_str = resample_params_to_str(dataset_job['rebuild_params']) if dataset_job['rebuild_params'] else "default"
    logger.log(f"🫠 Dataset Job: {dataset_job['experiment_name']}, {dataset_job['train_dataset']}, L{dataset_job['layer']}, {dataset_job['component']}, rebuild={rebuild_str}, seed={dataset_job['seed']}")

    try:
        # Get model info without loading the model
        model_name = config['model_name']
        d_model = get_model_d_model(model_name)  # Import this function
        device = config.get("device")
        
        # Validate train dataset for this specific seed
        try:
            logger.log(f"Validating train dataset '{dataset_job['train_dataset']}' with seed {dataset_job['seed']}")
            if should_skip_dataset(dataset_job['train_dataset'], data=None, logger=logger):
                logger.log(f"  - Skipping dataset job due to train dataset validation failure")
                return None # Return None to indicate skipping
        except Exception as e:
            logger.log(f"  - 💀 ERROR validating train dataset '{dataset_job['train_dataset']}' with seed {dataset_job['seed']}: {e}")
            return None # Return None to indicate skipping
        
        # Create seed-specific directory structure
        seed_dir = dataset_job['results_dir'] / f"seed_{dataset_job['seed']}"
        experiment_dir = seed_dir / dataset_job['experiment_name']
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        metric = dataset_job['experiment'].get('metric', 'acc')
        score_options = dataset_job['experiment'].get('score', ['all'])
        
        # Get effective seed for this rebuild_config
        effective_seed = get_effective_seed_for_rebuild_config(dataset_job['seed'], dataset_job['rebuild_params'])
        
        # Process all architectures for this dataset job
        logger.log(f"  🥰 Processing {len(dataset_job['architectures'])} architectures for dataset '{dataset_job['train_dataset']}'")
        
        all_eval_results = {}
        
        # Separate trainable and non-trainable probes
        trainable_architectures = []
        non_trainable_architectures = []
        
        for arch_config in dataset_job['architectures']:
            architecture_name = arch_config['name']
            config_name = arch_config.get('config_name')
            
            # Check if this is a non-trainable probe
            if (config_name.startswith('act_sim') or 
                config_name == 'mass_mean'):
                non_trainable_architectures.append(arch_config)
            else:
                trainable_architectures.append(arch_config)
        
        # Process non-trainable probes first
        for arch_config in non_trainable_architectures:
            architecture_name = arch_config['name']
            config_name = arch_config.get('config_name')
            
            logger.log(f"    😋 Running non-trainable probe: {architecture_name}, {config_name}")
            
            # Run non-trainable probe
            run_non_trainable_probe(
                model_name=model_name, d_model=d_model, train_dataset_name=dataset_job['train_dataset'],
                layer=dataset_job['layer'], component=dataset_job['component'], 
                architecture_name=architecture_name, 
                config_name=config_name,
                device=config['device'], use_cache=config['cache_activations'], 
                seed=effective_seed,
                results_dir=experiment_dir, cache_dir=dataset_job['cache_dir'], logger=logger, 
                retrain=retrain,
                rebuild_config=dataset_job['rebuild_params'], metric=metric,
                train_size=0.85,
                val_size=0,
                test_size=0.15
            )
        
        # Process trainable probes individually
        for arch_config in trainable_architectures:
            architecture_name = arch_config['name']
            config_name = arch_config.get('config_name')
            
            logger.log(f"    🚀 Training probe: {architecture_name}, {config_name}")
            
            # Train the probe
            train_probe(
                model_name=model_name, d_model=d_model, train_dataset_name=dataset_job['train_dataset'],
                layer=dataset_job['layer'], component=dataset_job['component'], 
                architecture_name=architecture_name, 
                config_name=config_name,
                device=config['device'], use_cache=config['cache_activations'], 
                seed=effective_seed,
                results_dir=experiment_dir, cache_dir=dataset_job['cache_dir'], logger=logger, 
                retrain=retrain,
                hyperparameter_tuning=hyperparameter_tuning, 
                rebuild_config=dataset_job['rebuild_params'], metric=metric,
                retrain_with_best_hparams=retrain_with_best_hparams,
                train_size=0.85,
                val_size=0,
                test_size=0.15
            )
        
        # Evaluate all probes on all evaluation datasets
        logger.log(f"    🤔 Evaluating all probes on {len(dataset_job['eval_datasets'])} datasets...")
        
        # Collect all architectures to evaluate
        all_architectures_to_evaluate = non_trainable_architectures + trainable_architectures
        
        for eval_dataset in dataset_job['eval_datasets']:
            try:
                # Validate eval dataset for this specific seed
                logger.log(f"      Validating eval dataset '{eval_dataset}' with seed {dataset_job['seed']}")
                if should_skip_dataset(eval_dataset, data=None, logger=logger):
                    logger.log(f"        - Skipping evaluation due to eval dataset validation failure")
                    continue
                                
            except Exception as e:
                logger.log(f"        - 💀 ERROR validating eval dataset '{eval_dataset}' with seed {dataset_job['seed']}: {e}")
                continue
            
            # Check if probe directory exists
            if dataset_job['rebuild_params'] is not None:
                probe_save_dir = experiment_dir / f"dataclass_exps_{dataset_job['train_dataset']}"
            else:
                probe_save_dir = experiment_dir / f"train_{dataset_job['train_dataset']}"
            
            if not probe_save_dir.exists():
                logger.log(f"        - [SKIP] Probe dir does not exist: {probe_save_dir}")
                continue
            
            # Evaluate each probe
            for arch_config in all_architectures_to_evaluate:
                architecture_name = arch_config['name']
                config_name = arch_config.get('config_name')
                
                logger.log(f"🤔 Evaluation of {architecture_name} ({config_name}) on {eval_dataset}")
                
                try:                        
                    metrics = evaluate_probe(
                        train_dataset_name=dataset_job['train_dataset'], 
                        eval_dataset_name=eval_dataset,
                        layer=dataset_job['layer'], component=dataset_job['component'], 
                        architecture_config=arch_config,
                        results_dir=experiment_dir, logger=logger, seed=effective_seed,
                        model_name=config['model_name'], d_model=d_model, device=config['device'],
                        use_cache=config['cache_activations'], cache_dir=dataset_job['cache_dir'], 
                        reevaluate=reevaluate,
                        score_options=score_options, rebuild_config=dataset_job['rebuild_params'], 
                        return_metrics=True,
                        train_size=0.85,
                        val_size=0,
                        test_size=0.15
                    )
                    
                    # Store results by rebuild config and eval dataset
                    rebuild_key = resample_params_to_str(dataset_job['rebuild_params'])
                    if rebuild_key not in all_eval_results:
                        all_eval_results[rebuild_key] = {}
                    if eval_dataset not in all_eval_results[rebuild_key]:
                        all_eval_results[rebuild_key][eval_dataset] = {}
                    
                    # Store results by architecture
                    all_eval_results[rebuild_key][eval_dataset][architecture_name] = metrics
                    
                except Exception as e:
                    logger.log(f"          - 💀 ERROR evaluating probe on '{eval_dataset}': {e}")
                    continue
        
        # Save evaluation results for this dataset job
        if all_eval_results:
            # Create a comprehensive filename that includes all architectures
            arch_names = [arch['name'] for arch in dataset_job['architectures']]
            arch_str = "_".join(arch_names[:3]) + ("_and_more" if len(arch_names) > 3 else "")
            
            # Determine the correct directory to save results
            if dataset_job['rebuild_params'] is not None:
                eval_results_path = experiment_dir / f"dataclass_exps_{dataset_job['train_dataset']}" / f"eval_on_all__{dataset_job['train_dataset']}_L{dataset_job['layer']}_{dataset_job['component']}_{arch_str}_allres.json"
                eval_results_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                eval_results_path = experiment_dir / f"train_{dataset_job['train_dataset']}" / f"eval_on_all__{dataset_job['train_dataset']}_L{dataset_job['layer']}_{dataset_job['component']}_{arch_str}_allres.json"
                eval_results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(eval_results_path, "w") as f:
                json.dump(all_eval_results, f, indent=2)
            
            logger.log(f"  🤗 Saved evaluation results to {eval_results_path}")
        
        logger.log(f"  😜 Completed dataset job")
        return all_eval_results # Return results for joblib to collect
        
    except Exception as e:
        logger.log(f"  💀 ERROR in dataset job: {e}")
        return None
    finally:
        # Clean up model to free GPU memory
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.close()

def main():
    # Clear GPU memory and set device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete

    global config_yaml, retrain, reevaluate, hyperparameter_tuning, retrain_with_best_hparams

    # Load config (already loaded at top, but reload to be safe)
    try:
        with open(f"configs/{config_yaml}", "r") as f:
            config = yaml.safe_load(f)
        # Re-apply device fix if needed
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
            check_on = check['check_on']
            # Support both single dataset and list of datasets
            if isinstance(check_on, list):
                datasets_to_check = check_on
            else:
                datasets_to_check = [check_on] if check_on else []
            
            # Check if all datasets have runthrough directories
            for ds_name in datasets_to_check:
                runthrough_dir = Path(f"results/{run_name}/runthrough_{ds_name}")
                if not runthrough_dir.exists():
                    all_checks_done = False
                    break
            if not all_checks_done:
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
        if config['model_name'] == "meta-llama/Llama-3.3-70B-Instruct": # quantize large model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",      # NormalFloat4, better for LLMs
                bnb_4bit_use_double_quant=True, # Second quantization for smaller memory footprint
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'],
                device_map=config['device'],
                quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'], 
                device_map=config['device'], 
                torch_dtype=torch.float16
            )
            
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        d_model = get_model_d_model(config['model_name'])

        available_datasets = get_available_datasets()

        # Step 1: Pre-flight check and gather all dataset jobs
        logger.log("\n Performing Pre-flight Checks and Gathering Dataset Jobs")
        all_dataset_jobs = []
        train_and_eval_datasets = set()

        for experiment in config['experiments']:
            experiment_name = experiment['name']
            train_sets = [experiment['train_on']]
            eval_sets = experiment['evaluate_on']
            
            train_and_eval_datasets.update(d for d in train_sets if d in available_datasets)
            train_and_eval_datasets.update(d for d in eval_sets if d in available_datasets)
            
            # Handle rebuild configs for this experiment
            rebuild_configs = []
            if experiment and 'rebuild_config' in experiment:
                rc = experiment['rebuild_config']
                if isinstance(rc, dict):
                    for group in rc.values():
                        rebuild_configs.extend(group)
                else:
                    rebuild_configs.extend(rc if isinstance(rc, list) else [rc])
            else:
                rebuild_configs = [None]
            
            for train_dataset in train_sets:
                for rebuild_params in rebuild_configs:
                    for layer in config['layers']:
                        for component in config['components']:
                            for seed in all_seeds:
                                # Create a dataset job that will process all architectures for this dataset/rebuild/seed combination
                                dataset_job = {
                                    'experiment_name': experiment_name,
                                    'train_dataset': train_dataset,
                                    'layer': layer,
                                    'component': component,
                                    'seed': seed,
                                    'rebuild_params': rebuild_params,
                                    'experiment': experiment,
                                    'eval_datasets': eval_sets,
                                    'architectures': config.get('architectures', []),
                                    'results_dir': results_dir,
                                    'cache_dir': cache_dir,
                                    'log_file_path': log_file_path, # Pass log file path instead of logger
                                }
                                all_dataset_jobs.append(dataset_job)

        # Step 2: Extract activations for train and eval datasets (optional but recommended)
        # Just extract them all even if they're a bit extra
        logger.log("\n" + "="*25 + " ACTIVATION EXTRACTION PHASE " + "="*25)
        for dataset_name in train_and_eval_datasets:
            for layer in config['layers']:
                for component in config['components']:
                    # Seed doesn't matter here, we're extracting over the whole dataset (even parts we wont use)
                    extract_activations_for_dataset_with_llm(model, tokenizer, config['model_name'], dataset_name, layer, component, device, all_seeds[0], logger, include_llm_samples=True, run_name=run_name)

        # Unload model after activation extraction to free GPU memory
        logger.log("\n" + "="*25 + " MODEL UNLOADING PHASE " + "="*25)
        logger.log("Unloading model to free GPU memory for parallel processing...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.log("Model unloaded successfully")

        # Step 3: Process each dataset job (train and evaluate all probes on one dataset)
        logger.log("\n" + "="*25 + " DATASET PROCESSING PHASE " + "="*25)
        logger.log(f"Processing {len(all_dataset_jobs)} dataset jobs...")
        
        # Use all available cores for maximum parallelization
        n_jobs = min(2, len(all_dataset_jobs))  # Use up to 32 cores (conservative for memory)
        logger.log(f"Using {n_jobs} parallel jobs")
        
        all_eval_results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_dataset_job)(
                dataset_job, config, retrain, hyperparameter_tuning, retrain_with_best_hparams, reevaluate
            )
            for dataset_job in all_dataset_jobs
        )

        # Combine results from all jobs
        all_eval_results = {}
        completed_jobs = 0
        for results_dict in all_eval_results_list:
            if results_dict: # Only add if not skipped
                all_eval_results.update(results_dict)
                completed_jobs += 1
        
        logger.log(f"Completed {completed_jobs}/{len(all_dataset_jobs)} dataset jobs successfully")
        
        # Save combined evaluation results if any jobs completed
        if all_eval_results:
            logger.log("Combining and saving evaluation results...")
            # Note: Individual job results are already saved by each process
            # This is just for logging purposes
                
    finally:
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.log("=" * 80)
        logger.log("\U0001f979 Run finished. Closing log file.")
        logger.close()

if __name__ == "__main__":
    main()
