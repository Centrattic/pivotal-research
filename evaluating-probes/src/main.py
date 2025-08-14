import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import json
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from joblib import Parallel, delayed
import sys
import os
from configs.probes import PROBE_CONFIGS, SklearnLinearProbeConfig, PytorchAttentionProbeConfig, SAEProbeConfig, MassMeanProbeConfig, ActivationSimilarityProbeConfig
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Set CUDA device BEFORE importing transformer_lens
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument('--rerun', action='store_true', help='Rerun all probes from scratch, bypassing cached results')

args = parser.parse_args()
global config_yaml, rerun
config_yaml = args.config + "_config.yaml"
rerun = args.rerun

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

# Now import after setting CUDA device
from src.data import Dataset, get_available_datasets, get_model_d_model
from src.utils_training import train_single_probe, evaluate_single_probe, extract_activations_for_dataset
from src.logger import Logger
from src.utils import resample_params_to_str, get_effective_seeds, generate_llm_upsampling_configs
from src.data import get_model_d_model
from src.model_check.main import run_model_check
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ProbeJob:
    """Represents a single probe training/evaluation job."""
    # Basic identifiers
    experiment_name: str
    train_dataset: str
    eval_datasets: List[str]
    layer: int
    component: str
    seed: int

    # Probe configuration
    architecture_name: str  # sklearn_linear, linear, sae, attention, act_sim, mass_mean
    probe_config: Any  # Union[SklearnLinearProbeConfig, PytorchAttentionProbeConfig, SAEProbeConfig, MassMeanProbeConfig, ActivationSimilarityProbeConfig]

    # Data configuration
    rebuild_config: Optional[Dict] = None
    on_policy: bool = False  # Whether this is an on-policy probe

    # Training configuration
    train_size: float = 0.85
    val_size: float = 0.0
    test_size: float = 0.15

    # Configuration for activation extraction and model check
    format_type: str = "r-no-it"  # Activation extraction format
    config: Optional[Dict] = None  # Full config for access to activation_extraction settings


def create_probe_config(
    architecture_name: str, config_name: str
) -> Union[SklearnLinearProbeConfig, PytorchAttentionProbeConfig, SAEProbeConfig, MassMeanProbeConfig,
           ActivationSimilarityProbeConfig]:
    """Create probe configuration based on architecture and config name."""
    # Use the existing PROBE_CONFIGS dictionary
    if config_name in PROBE_CONFIGS:
        return PROBE_CONFIGS[config_name]
    else:
        # Fallback to creating configs based on architecture and config name
        if architecture_name == "sklearn_linear":
            aggregation = config_name.split("_")[-1] if "_" in config_name else "mean"
            return SklearnLinearProbeConfig(aggregation=aggregation)
        elif architecture_name == "sae":
            # Handle specific SAE configs
            if "16k_l0_408" in config_name:
                return SAEProbeConfig(sae_id="layer_20/width_16k/average_l0_408", aggregation="last")
            elif "262k_l0_259" in config_name:
                return SAEProbeConfig(sae_id="layer_20/width_262k/average_l0_259", aggregation="last")
            else:
                aggregation = config_name.split("_")[-1] if "_" in config_name else "mean"
                return SAEProbeConfig(aggregation=aggregation)
        elif architecture_name == "attention":
            return PytorchAttentionProbeConfig()
        elif architecture_name == "act_sim":
            aggregation = config_name.split("_")[-1] if "_" in config_name else "mean"
            return ActivationSimilarityProbeConfig(aggregation=aggregation)
        elif architecture_name == "mass_mean":
            return MassMeanProbeConfig(use_iid=False)
        else:
            raise ValueError(f"Unknown architecture: {architecture_name}")


def add_hyperparams_to_filename(
    base_filename: str,
    probe_config,
) -> str:
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


def generate_hyperparameter_sweep(
    architecture_name: str, config_name: str
) -> List[Union[SklearnLinearProbeConfig, PytorchAttentionProbeConfig, SAEProbeConfig, MassMeanProbeConfig,
                ActivationSimilarityProbeConfig]]:
    """Generate hyperparameter sweep configurations for a given architecture."""
    # Define sweep arrays at the top for easy editing
    C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
    LR_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    WEIGHT_DECAY_VALUES = [0.0, 1e-5, 1e-4, 1e-3]

    sweep_configs = []

    if architecture_name == "sklearn_linear":
        # C sweep
        for C in C_VALUES:
            base_config = create_probe_config(architecture_name, config_name)
            base_config.C = C
            sweep_configs.append(base_config)
    elif architecture_name in ["sae", "attention"]:
        # Learning rate sweep for trainable architectures
        for lr in LR_VALUES:
            base_config = create_probe_config(architecture_name, config_name)
            base_config.lr = lr
            sweep_configs.append(base_config)
    else:
        # Non-trainable probes don't need hyperparameter sweeps
        sweep_configs = [create_probe_config(architecture_name, config_name)]

    return sweep_configs


def run_model_checks(
    config: Dict,
    logger: Logger,
):
    """Run model checks if specified in config."""
    if 'model_check' not in config:
        return

    # Check if any experiments are on-policy
    has_on_policy = False
    for experiment in config['experiments']:
        if experiment.get('on_policy', False):
            has_on_policy = True
            break

    # Don't run model_check for on-policy experiments
    if has_on_policy:
        logger.log("\n=== Skipping model_check: on-policy experiments detected ===\n")
        return

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


def run_llm_upsampling(
    config: Dict,
    logger: Logger,
):
    """Run LLM upsampling if specified in config."""
    # Check if any experiments have LLM upsampling
    has_llm_upsampling = False
    for experiment in config['experiments']:
        if 'rebuild_config' in experiment:
            rc = experiment['rebuild_config']
            if isinstance(rc, dict):
                for group in rc.values():
                    if isinstance(group, list):
                        for item in group:
                            if isinstance(item, dict) and item.get('llm_upsampling', False):
                                has_llm_upsampling = True
                                break
                    elif isinstance(group, dict) and group.get('llm_upsampling', False):
                        has_llm_upsampling = True
                        break
            elif isinstance(rc, list):
                for item in rc:
                    if isinstance(item, dict) and item.get('llm_upsampling', False):
                        has_llm_upsampling = True
                        break

    if not has_llm_upsampling:
        logger.log("No LLM upsampling experiments found, skipping LLM upsampling")
        return

    logger.log("\n=== Running LLM upsampling ===")

    # Import and run the LLM upsampling script
    try:
        from src.llm_upsampling.llm_upsampling_script import run_llm_upsampling as run_llm_script

        # Get API key from environment or config
        api_key = config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logger.log("Warning: No OpenAI API key found. LLM upsampling will be skipped.")
            return

        # Run LLM upsampling
        config_path = Path(f"configs/{config_yaml}")
        run_llm_script(config_path, api_key)
        logger.log("=== LLM upsampling complete ===")

    except Exception as e:
        logger.log(f"Error during LLM upsampling: {e}")
        logger.log("Continuing without LLM upsampling...")


def extract_all_activations(
    config: Dict,
    model,
    tokenizer,
    logger: Logger,
):
    """Extract activations for all datasets, layers, and components."""
    logger.log("\n" + "=" * 25 + " ACTIVATION EXTRACTION PHASE " + "=" * 25)

    # Get all datasets that will be used
    all_datasets = set()
    for experiment in config['experiments']:
        train_sets = [experiment['train_on']]
        eval_sets = experiment['evaluate_on']
        all_datasets.update(train_sets)
        all_datasets.update(eval_sets)

        # Extract activations for each dataset, layer, and component
    for dataset_name in all_datasets:
        for layer in config['layers']:
            for component in config['components']:
                # Check if this dataset is used in any on-policy experiments
                on_policy = False
                for experiment in config['experiments']:
                    if experiment.get('on_policy', False):
                        train_sets = [experiment['train_on']]
                        eval_sets = experiment['evaluate_on']
                        if dataset_name in train_sets or dataset_name in eval_sets:
                            on_policy = True
                            break

                # Get activation extraction format from config
                if 'activation_extraction' not in config:
                    raise ValueError(
                        "activation_extraction.format_type must be specified in config. Options: 'qr' (on-policy), 'r' (off-policy instruct), 'r-no-it' (off-policy non-instruct)"
                    )
                format_type = config['activation_extraction']['format_type']
                if format_type not in ["qr", "r", "r-no-it"]:
                    raise ValueError(f"Invalid format_type: {format_type}. Must be one of: 'qr', 'r', 'r-no-it'")

                extract_activations_for_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=config['model_name'],
                    dataset_name=dataset_name,
                    layer=layer,
                    component=component,
                    device=config['device'],
                    seed=all_seeds[0],  # Use first seed for activation extraction
                    logger=logger,
                    on_policy=on_policy,
                    include_llm_samples=True,
                    format_type=format_type
                )


def create_probe_jobs(config: Dict, all_seeds: List[int]) -> List[ProbeJob]:
    """Create all probe jobs from the configuration."""
    all_jobs = []

    for experiment in config['experiments']:
        experiment_name = experiment['name']
        train_sets = [experiment['train_on']]
        eval_sets = experiment['evaluate_on']
        on_policy = experiment.get('on_policy', False)

        # Handle rebuild configs
        rebuild_configs = []
        if 'rebuild_config' in experiment:
            rc = experiment['rebuild_config']
            if isinstance(rc, dict):
                for group in rc.values():
                    rebuild_configs.extend(group)
            else:
                rebuild_configs.extend(rc if isinstance(rc, list) else [rc])
        else:
            rebuild_configs = [None]

        # Generate hyperparameter sweeps for each architecture
        for arch_config in config.get('architectures', []):
            architecture_name = arch_config['name']
            config_name = arch_config.get('config_name')
            # Generate hyperparameter sweep (always use predefined sweeps)
            probe_configs = generate_hyperparameter_sweep(architecture_name, config_name)

            for probe_config in probe_configs:
                for train_dataset in train_sets:
                    for rebuild_params in rebuild_configs:
                        for layer in config['layers']:
                            for component in config['components']:
                                for seed in all_seeds:
                                    # Get activation extraction format from config
                                    if 'activation_extraction' not in config:
                                        raise ValueError(
                                            "activation_extraction.format_type must be specified in config. Options: 'qr' (on-policy), 'r' (off-policy instruct), 'r-no-it' (off-policy non-instruct)"
                                        )
                                    format_type = config['activation_extraction']['format_type']
                                    if format_type not in ["qr", "r", "r-no-it"]:
                                        raise ValueError(
                                            f"Invalid format_type: {format_type}. Must be one of: 'qr', 'r', 'r-no-it'"
                                        )

                                    job = ProbeJob(
                                        experiment_name=experiment_name,
                                        train_dataset=train_dataset,
                                        eval_datasets=eval_sets,
                                        layer=layer,
                                        component=component,
                                        seed=seed,
                                        architecture_name=architecture_name,
                                        probe_config=probe_config,
                                        rebuild_config=rebuild_params,
                                        on_policy=on_policy,
                                        format_type=format_type,
                                        config=config
                                    )
                                    all_jobs.append(job)

    return all_jobs


def process_probe_job(
    job: ProbeJob,
    config: Dict,
    results_dir: Path,
    cache_dir: Path,
    log_file_path: Path,
):
    """Process a single probe job (train and evaluate)."""
    # Create a separate logger for this process
    logger = Logger(log_file_path)

    logger.log("-" * 80)
    rebuild_str = resample_params_to_str(job.rebuild_config) if job.rebuild_config else "default"
    logger.log(
        f"🫠 Probe Job: {job.experiment_name}, {job.train_dataset}, L{job.layer}, {job.component}, rebuild={rebuild_str}, seed={job.seed}"
    )

    try:
        # Create seed-specific directory structure
        seed_dir = results_dir / f"seed_{job.seed}"
        experiment_dir = seed_dir / job.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories within experiment directory
        trained_dir = experiment_dir / "trained"
        val_eval_dir = experiment_dir / "val_eval"
        test_eval_dir = experiment_dir / "test_eval"
        gen_eval_dir = experiment_dir / "gen_eval"

        for dir_path in [trained_dir, val_eval_dir, test_eval_dir, gen_eval_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Train the probe
        logger.log(f"  🚀 Training probe: {job.architecture_name}")
        train_single_probe(
            job=job, config=config, results_dir=trained_dir, cache_dir=cache_dir, logger=logger, rerun=rerun
        )

        # Evaluate on all evaluation datasets
        logger.log(f"  🤔 Evaluating probe on {len(job.eval_datasets)} datasets...")

        for eval_dataset in job.eval_datasets:
            # Determine which evaluation directory to use
            if eval_dataset == job.train_dataset:
                # Same dataset - evaluate on both validation and test sets
                # First evaluate on validation set
                logger.log(f"    Evaluating on {eval_dataset} validation set")
                try:
                    metrics = evaluate_single_probe(
                        job=job,
                        eval_dataset=eval_dataset,
                        config=config,
                        results_dir=val_eval_dir,
                        cache_dir=cache_dir,
                        logger=logger,
                        only_test=False,  # Use validation set
                        rerun=rerun
                    )
                    logger.log(f"    ✅ Validation evaluation complete for {eval_dataset}")
                except Exception as e:
                    logger.log(f"    💀 ERROR evaluating validation on '{eval_dataset}': {e}")

                # Then evaluate on test set
                logger.log(f"    Evaluating on {eval_dataset} test set")
                eval_dir = test_eval_dir
                only_test = True
            else:
                # Different dataset - use gen_eval with only_test=True
                eval_dir = gen_eval_dir
                only_test = True

            logger.log(f"    Evaluating on {eval_dataset} (only_test={only_test})")

            try:
                metrics = evaluate_single_probe(
                    job=job,
                    eval_dataset=eval_dataset,
                    config=config,
                    results_dir=eval_dir,
                    cache_dir=cache_dir,
                    logger=logger,
                    only_test=only_test,
                    rerun=rerun
                )

                logger.log(f"    ✅ Evaluation complete for {eval_dataset}")

            except Exception as e:
                logger.log(f"    💀 ERROR evaluating on '{eval_dataset}': {e}")
                continue

        logger.log(f"  😜 Completed probe job")
        return True

    except Exception as e:
        logger.log(f"  💀 ERROR in probe job: {e}")
        return False
    finally:
        # Clean up model to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.close()


def main():
    # Clear GPU memory and set device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    global config_yaml, rerun

    # Load config
    try:
        with open(f"configs/{config_yaml}", "r") as f:
            config = yaml.safe_load(f)
        # Re-apply device fix if needed
        device = config.get("device")
        if device and "cuda" in device and "1" in device:
            config["device"] = "cuda:0"
            print(f"Updated device from {device} to cuda:0")
    except:
        print(f"A config of name {config_yaml} does not exist.")
        return

    run_name = config.get('run_name', 'default_run')
    results_dir = Path("results") / run_name
    cache_dir = Path("activation_cache") / config['model_name']
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get log file path from config
    log_file_path = config.get('log_file', results_dir / "output.log")
    if isinstance(log_file_path, str):
        log_file_path = Path(log_file_path)
        if not log_file_path.is_absolute():
            log_file_path = results_dir / log_file_path

    # Ensure the log file directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = Logger(log_file_path)

    # Get all seeds to process
    all_seeds = config.get('seeds', [42])
    if not isinstance(all_seeds, list):
        all_seeds = [all_seeds]
    logger.log(f"Processing {len(all_seeds)} seeds: {all_seeds}")

    # Step 1: Run model checks
    run_model_checks(config, logger)

    # Step 2: Run LLM upsampling if needed
    run_llm_upsampling(config, logger)

    try:
        # Step 3: Load model for activation extraction
        logger.log(f"Loading model '{config['model_name']}' for activation extraction...")
        device = config.get("device")
        if config['model_name'] == "meta-llama/Llama-3.3-70B-Instruct":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'], device_map=config['device'], quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config['model_name'], device_map=config['device'], torch_dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Step 4: Ensures activations are extracted
        extract_all_activations(config, model, tokenizer, logger)

        # Unload model after activation extraction to free GPU memory
        logger.log("\n" + "=" * 25 + " MODEL UNLOADING PHASE " + "=" * 25)
        logger.log("Unloading model to free GPU memory for parallel processing...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.log("Model unloaded successfully")

        # Step 5: Create all probe jobs
        logger.log("\n" + "=" * 25 + " JOB CREATION PHASE " + "=" * 25)
        all_probe_jobs = create_probe_jobs(config, all_seeds)
        logger.log(f"Created {len(all_probe_jobs)} probe jobs")

        # Step 6: Process all probe jobs in parallel
        logger.log("\n" + "=" * 25 + " PROBE PROCESSING PHASE " + "=" * 25)
        logger.log(f"Processing {len(all_probe_jobs)} probe jobs...")

        # Use all available cores for maximum parallelization
        n_jobs = min(40, len(all_probe_jobs))  # Conservative for memory
        logger.log(f"Using {n_jobs} parallel jobs")

        results = Parallel(
            n_jobs=n_jobs, verbose=10
        )(delayed(process_probe_job)(job, config, results_dir, cache_dir, log_file_path) for job in all_probe_jobs)

        # Count successful jobs
        successful_jobs = sum(1 for result in results if result)
        logger.log(f"Completed {successful_jobs}/{len(all_probe_jobs)} probe jobs successfully")

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
