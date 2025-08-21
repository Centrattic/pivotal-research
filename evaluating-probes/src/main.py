import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import json
import copy
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict
from joblib import Parallel, delayed
import sys
import subprocess
import os
from dotenv import load_dotenv, find_dotenv
from configs.probes import (
    PROBE_CONFIGS,
    ProbeJob,
    SklearnLinearProbeConfig,
    PytorchAttentionProbeConfig,
    SAEProbeConfig,
    MassMeanProbeConfig,
    ActivationSimilarityProbeConfig,
)
from dataclasses import dataclass

# Load environment variables from a .env file if present (prefer repo root)
repo_root = Path(
    __file__,
).resolve().parents[1]
env_path = repo_root / ".env"
if env_path.exists():
    load_dotenv(
        env_path,
    )
    print(
        f"Loaded environment variables from {env_path}",
    )
else:
    raise FileNotFoundError(
        f"No .env file found in {repo_root}",
    )


# Debug helper to print selected environment variables (masking secrets)
def _mask_secret(
    value: Optional[str],
    keep_last: int = 4,
) -> str:
    if not value:
        return "None"
    if len(value, ) <= keep_last:
        return "*" * len(
            value,
        )
    return f"{value[:2]}{'*' * max(0, len(value,) - (2 + keep_last),)}{value[-keep_last:]}"


def _print_env_vars():
    keys = ["OPENAI_API_KEY", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "EP_CPU_NJOBS", "EP_GPU_NJOBS"]
    print(
        "\n=== DEBUG: Environment Variables ===",
    )
    for key in keys:
        value = os.environ.get(
            key,
        )
        if key == "OPENAI_API_KEY":
            print(
                f"{key}={_mask_secret(value,)}",
            )
        else:
            print(
                f"{key}={value}",
            )
    print(
        "====================================\n",
    )


_print_env_vars()

# Set CUDA device BEFORE importing transformer_lens
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
)
parser.add_argument(
    '--rerun',
    action='store_true',
    help='Rerun all probes from scratch, bypassing cached results',
)

args = parser.parse_args()
global config_yaml, rerun
config_yaml = args.config + "_config.yaml"
rerun = args.rerun

# Load config early to set CUDA device before any CUDA operations
try:
    with open(
            f"configs/{config_yaml}",
            "r",
    ) as f:
        config = yaml.safe_load(
            f,
        )
    device = config.get(
        "device",
    )
    if device and "cuda" in device:
        cuda_id = device.split(
            ":",
        )[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
        # Update the device string to use cuda:0 since we're now only showing one device
        config["device"] = "cuda:0"
        print(
            f"Set CUDA_VISIBLE_DEVICES to {cuda_id}, updated device to cuda:0",
        )
except Exception as e:
    print(
        f"Warning: Could not set CUDA device early: {e}",
    )

# Print debug env after potential CUDA changes
_print_env_vars()

# Now import after setting CUDA device
from src.data import Dataset, get_available_datasets, get_model_d_model
from src.utils_training import train_single_probe, evaluate_single_probe, extract_activations_for_dataset
from src.logger import Logger
from src.utils import resample_params_to_str
from src.model_check.main import run_model_check
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    # Keep BLAS/OMP-backed libs on a limited number of threads
    from threadpoolctl import threadpool_limits  # type: ignore
    # Allow override via environment variable, default to 1 for CPU-intensive ops
    cpu_jobs = int(os.environ.get("EP_CPU_NJOBS", "15"))
    # For process parallelism, we want to limit threads within each process
    # If cpu_jobs > 1, it will be used for process parallelism, so limit threads to 1
    thread_limit = 1 if cpu_jobs > 1 else cpu_jobs
    threadpool_limits(
        thread_limit,
    )
    # Also limit PyTorch intra-op and inter-op threads
    torch.set_num_threads(
        thread_limit,
    )
    torch.set_num_interop_threads(
        thread_limit,
    )
except Exception:
    pass

# Configure GPU job limits
try:
    # Set CUDA job limits for GPU operations
    # Allow override via environment variable, default to 1 for GPU-intensive ops
    gpu_jobs = int(os.environ.get("EP_GPU_NJOBS", "1"))
    if torch.cuda.is_available():
        # Set CUDA device properties to limit concurrent operations
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            # Note: PyTorch doesn't have direct thread control for CUDA,
            # but we can set environment variables that affect CUDA concurrency
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1" if gpu_jobs == 1 else "0"
except Exception:
    pass


def create_probe_config(
    architecture_name: str,
    config_name: str,
) -> Union[SklearnLinearProbeConfig, PytorchAttentionProbeConfig, SAEProbeConfig, MassMeanProbeConfig,
           ActivationSimilarityProbeConfig]:
    """Create probe configuration based on architecture and config name."""
    # Use the existing PROBE_CONFIGS dictionary
    if config_name in PROBE_CONFIGS:
        # Return a deepcopy so callers can safely mutate fields (e.g., C in sweeps)
        return copy.deepcopy(
            PROBE_CONFIGS[config_name],
        )
    else:
        # Fallback to creating configs based on architecture and config name
        if architecture_name == "sklearn_linear":
            aggregation = config_name.split(
                "_",
            )[-1] if "_" in config_name else "mean"
            return SklearnLinearProbeConfig(
                aggregation=aggregation,
            )
        elif architecture_name == "sae":
            # Handle specific SAE configs
            if "16k_l0_408" in config_name:
                return SAEProbeConfig(
                    sae_id="layer_20/width_16k/average_l0_408",
                    aggregation="last",
                )
            elif "262k_l0_259" in config_name:
                return SAEProbeConfig(
                    sae_id="layer_20/width_262k/average_l0_259",
                    aggregation="last",
                )
            else:
                aggregation = config_name.split(
                    "_",
                )[-1] if "_" in config_name else "mean"
                return SAEProbeConfig(
                    aggregation=aggregation,
                )
        elif architecture_name == "attention":
            return PytorchAttentionProbeConfig()
        elif architecture_name == "act_sim":
            aggregation = config_name.split(
                "_",
            )[-1] if "_" in config_name else "mean"
            return ActivationSimilarityProbeConfig(
                aggregation=aggregation,
            )
        elif architecture_name == "mass_mean":
            return MassMeanProbeConfig(
                use_iid=False,
            )
        else:
            raise ValueError(
                f"Unknown architecture: {architecture_name}",
            )


def add_hyperparams_to_filename(
    base_filename: str,
    probe_config,
) -> str:
    """Add hyperparameter values to filename if they differ from defaults."""
    hparam_suffix = ""

    # Common hyperparameters to include
    hparams_to_check = [
        'C',
        'lr',
        'weight_decay',
        'batch_size',
        'epochs',
        'top_k_features',
    ]

    for hparam in hparams_to_check:
        if hasattr(
                probe_config,
                hparam,
        ):
            value = getattr(
                probe_config,
                hparam,
            )
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
            elif hparam == 'top_k_features':
                # Always include top-k for SAE sweeps to disambiguate artifacts
                hparam_suffix += f"_topk{value}"

    if hparam_suffix:
        return base_filename + hparam_suffix
    else:
        return base_filename


def generate_hyperparameter_sweep(
    architecture_name: str,
    config_name: str,
) -> List[Union[SklearnLinearProbeConfig, PytorchAttentionProbeConfig, SAEProbeConfig, MassMeanProbeConfig,
                ActivationSimilarityProbeConfig,]]:
    """Generate hyperparameter sweep configurations for a given architecture."""
    # there is a default filename omission rule: C=1, lr=5e-4, weight_decay=0.0
    # top_k=3584 always included in the filename as a default

    # Defaults
    # C_VALUES = [1.0]
    # LR_VALUES = [5e-4]
    # WEIGHT_DECAY_VALUES = [0.0]
    # TOP_K_VALUES = [3584]

    C_VALUES = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4] # all we have time for
    TOP_K_VALUES = [128, 256, 512, 1024, 2048, 3584, 4096, 8192]
    # Sweep over top k values
    # LR and weight decay handled by find_best_fit and attention

    sweep_configs = []

    if architecture_name == "sklearn_linear":
        # C sweep
        for C in C_VALUES:
            base_config = create_probe_config(
                architecture_name,
                config_name,
            )
            base_config.C = C
            sweep_configs.append(
                base_config,
            )
    elif architecture_name == "attention":
        # For attention probes, we only create the default config
        # The find_best_fit function will handle hyperparameter sweeping internally
        # This avoids creating multiple jobs and instead does efficient batch training
        sweep_configs = [create_probe_config(
            architecture_name,
            config_name,
        )]
    elif architecture_name == "sae" or architecture_name.startswith("sae", ):
        # SAE uses sklearn LogisticRegression; sweep C and number of selected SAE features
        for C in C_VALUES:
            for top_k in TOP_K_VALUES:
                base_config = create_probe_config(
                    "sae",
                    config_name,
                )
                base_config.C = C
                # Some configs (e.g., llama) may fix top_k; set if present
                if hasattr(
                        base_config,
                        'top_k_features',
                ):
                    base_config.top_k_features = top_k
                sweep_configs.append(
                    base_config,
                )
    else:
        # Non-trainable probes don't need hyperparameter sweeps
        sweep_configs = [create_probe_config(
            architecture_name,
            config_name,
        )]

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
        if experiment.get(
                'on_policy',
                False,
        ):
            has_on_policy = True
            break

    # Don't run model_check for on-policy experiments
    if has_on_policy:
        logger.log(
            "\n=== Skipping model_check: on-policy experiments detected ===\n",
        )
        return

    run_name = config.get(
        'run_name',
        'default_run',
    )
    all_checks_done = True

    for check in config['model_check']:
        check_on = check['check_on']
        # Support both single dataset and list of datasets
        if isinstance(
                check_on,
                list,
        ):
            datasets_to_check = check_on
        else:
            datasets_to_check = [check_on] if check_on else []

        # Check if all datasets have runthrough directories
        for ds_name in datasets_to_check:
            runthrough_dir = Path(
                f"results/{run_name}/runthrough_{ds_name}",
            )
            if not runthrough_dir.exists():
                all_checks_done = False
                break
        if not all_checks_done:
            break

    if not all_checks_done:
        logger.log(
            "\n=== Running model_check before main pipeline ===",
        )
        run_model_check(
            config,
            logger,
        )
        logger.log(
            "=== model_check complete ===\n",
        )
    else:
        logger.log(
            "\n=== Skipping model_check: all runthrough directories already exist ===\n",
        )


def run_llm_upsampling(
    config: Dict,
    logger: Logger,
):
    """Run LLM upsampling if specified in config."""
    # Check for llm_upsampling_experiments block in any experiment
    has_llm_upsampling = any(
        isinstance(
            experiment.get(
                'rebuild_config',
            ),
            dict,
        ) and bool(
            experiment['rebuild_config'].get(
                'llm_upsampling_experiments',
            ),
        ) for experiment in config.get(
            'experiments',
            [],
        )
    )

    if not has_llm_upsampling:
        logger.log(
            "No llm_upsampling_experiments found, skipping LLM upsampling",
        )
        return

    logger.log(
        "\n=== Running LLM upsampling (external script) ===",
    )

    # Call the external upsampling script via subprocess to keep behavior identical
    try:
        # Determine config argument expected by the script (-c takes base name or path)
        config_arg = config_yaml[:-12] if config_yaml.endswith(
            "_config.yaml",
        ) else config_yaml

        # Get API key from environment or config
        api_key = config.get(
            'openai_api_key',
        ) or os.environ.get(
            'OPENAI_API_KEY',
        )
        if not api_key:
            logger.log(
                "Warning: No OPENAI_API_KEY found in env or config; skipping LLM upsampling.",
            )
            return

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "src.llm_upsampling.llm_upsampling_script",
            "-c",
            str(
                config_arg,
            ),
            "--api-key",
            str(
                api_key,
            ),
        ]

        logger.log(
            f"Invoking: {' '.join(cmd,)}",
        )
        # Ensure we execute from repo root (evaluating-probes)
        subprocess.run(
            cmd,
            cwd=str(
                repo_root,
            ),
            check=True,
        )
        logger.log(
            "=== LLM upsampling complete ===",
        )

    except subprocess.CalledProcessError as e:
        logger.log(
            f"Error during LLM upsampling (subprocess failed with code {e.returncode}).",
        )
        logger.log(
            "Continuing without LLM upsampling...",
        )
    except Exception as e:
        logger.log(
            f"Error during LLM upsampling: {e}",
        )
        logger.log(
            "Continuing without LLM upsampling...",
        )


def extract_all_activations(
    config: Dict,
    model,
    tokenizer,
    logger: Logger,
    all_seeds: List[int],
    results_dir: Path,
):
    """Extract activations for all datasets, layers, and components."""
    logger.log(
        "\n" + "=" * 25 + " ACTIVATION EXTRACTION PHASE " + "=" * 25,
    )

    # Get all datasets that will be used
    all_datasets = set()
    for experiment in config['experiments']:
        train_sets = [experiment['train_on']]
        eval_sets = experiment['evaluate_on']
        all_datasets.update(
            train_sets,
        )
        all_datasets.update(
            eval_sets,
        )

        # Extract activations for each dataset, layer, and component
    for dataset_name in all_datasets:
        for layer in config['layers']:
            for component in config['components']:
                # Check if this dataset is used in any on-policy experiments
                on_policy = False
                for experiment in config['experiments']:
                    if experiment.get(
                            'on_policy',
                            False,
                    ):
                        train_sets = [experiment['train_on']]
                        eval_sets = experiment['evaluate_on']
                        if dataset_name in train_sets or dataset_name in eval_sets:
                            on_policy = True
                            break

                # Check if this dataset is used as a training set (only include LLM samples for training sets)
                is_training_set = False
                for experiment in config['experiments']:
                    train_sets = [experiment['train_on']]
                    if dataset_name in train_sets:
                        is_training_set = True
                        break

                # Get activation extraction format from config
                if 'activation_extraction' not in config:
                    raise ValueError(
                        "activation_extraction.format_type must be specified in config. Options: 'qr' (on-policy), 'r' (off-policy instruct), 'r-no-it' (off-policy non-instruct)",
                    )
                format_type = config['activation_extraction']['format_type']
                if format_type not in ["qr", "r", "r-no-it"]:
                    raise ValueError(
                        f"Invalid format_type: {format_type}. Must be one of: 'qr', 'r', 'r-no-it'",
                    )

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
                    include_llm_samples=is_training_set,  # Only include LLM samples for training sets
                    format_type=format_type,
                    results_dir=results_dir,
                )


def create_probe_jobs(
    config: Dict,
    all_seeds: List[int],
) -> List[ProbeJob]:
    """Create all probe jobs from the configuration."""
    all_jobs = []

    for experiment in config['experiments']:
        experiment_name = experiment['name']
        train_sets = [experiment['train_on']]
        eval_sets = experiment['evaluate_on']
        on_policy = experiment.get(
            'on_policy',
            False,
        )

        # Handle rebuild configs
        rebuild_configs = []
        if 'rebuild_config' in experiment:
            rc = experiment['rebuild_config']
            if isinstance(
                    rc,
                    dict,
            ):
                for group in rc.values():
                    rebuild_configs.extend(
                        group,
                    )
            else:
                rebuild_configs.extend(
                    rc if isinstance(
                        rc,
                        list,
                    ) else [rc],
                )
        else:
            rebuild_configs = [None]

        # Generate hyperparameter sweeps for each architecture
        for arch_config in config.get(
                'architectures',
            [],
        ):
            architecture_name = arch_config['name']
            config_name = arch_config.get(
                'config_name',
            )
            # Generate hyperparameter sweep (always use predefined sweeps)
            probe_configs = generate_hyperparameter_sweep(
                architecture_name,
                config_name,
            )

            for probe_config in probe_configs:
                for train_dataset in train_sets:
                    for rebuild_params in rebuild_configs:
                        for layer in config['layers']:
                            for component in config['components']:
                                for seed in all_seeds:
                                    # Get activation extraction format from config
                                    if 'activation_extraction' not in config:
                                        raise ValueError(
                                            "activation_extraction.format_type must be specified in config. Options: 'qr' (on-policy), 'r' (off-policy instruct), 'r-no-it' (off-policy non-instruct)",
                                        )
                                    format_type = config['activation_extraction']['format_type']
                                    if format_type not in ["qr", "r", "r-no-it"]:
                                        raise ValueError(
                                            f"Invalid format_type: {format_type}. Must be one of: 'qr', 'r', 'r-no-it'",
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
                                    )
                                    all_jobs.append(
                                        job,
                                    )

    return all_jobs


def _canonical_rebuild_key(
    rebuild_config: Optional[Dict],
) -> str:
    if rebuild_config is None:
        return "__none__"
    try:
        return json.dumps(
            rebuild_config,
            sort_keys=True,
        )
    except Exception:
        return str(
            rebuild_config,
        )


def _build_group_dataset(
    job: ProbeJob,
    config: Dict,
) -> Dataset:
    """Build a Dataset for a group key, mirroring logic in train_single_probe."""
    # Base dataset from train dataset
    if job.rebuild_config is not None:
        orig_ds = Dataset(
            job.train_dataset,
            model_name=config['model_name'],
            device=config['device'],
            seed=job.seed,
        )

        # Filter data for off-policy experiments if model_check was run
        if not job.on_policy and 'model_check' in config:
            run_name = config.get(
                'run_name',
                'default_run',
            )
            for check in config['model_check']:
                check_on = check['check_on']
                datasets_to_check = check_on if isinstance(
                    check_on,
                    list,
                ) else ([check_on] if check_on else [])
                if job.train_dataset in datasets_to_check:
                    orig_ds.filter_data_by_model_check(
                        run_name,
                        check['name'],
                    )
                    break

        # Rebuild variants
        if 'llm_upsampling' in job.rebuild_config and job.rebuild_config['llm_upsampling']:
            n_real_neg = job.rebuild_config.get(
                'n_real_neg',
            )
            n_real_pos = job.rebuild_config.get(
                'n_real_pos',
            )
            upsampling_factor = job.rebuild_config.get(
                'upsampling_factor',
            )

            results_path = Path(
                config.get(
                    'results_dir',
                    '.',
                ),
            )
            run_name = config.get(
                'run_name',
                results_path.name,
            )
            # Base path to results; Dataset helper will append seed and dataset
            llm_csv_base_path = Path(
                job.rebuild_config.get(
                    'llm_csv_base_path',
                    results_path,
                ),
            )

            ds = Dataset.build_llm_upsampled_dataset(
                orig_ds,
                seed=job.seed,
                n_real_neg=n_real_neg,
                n_real_pos=n_real_pos,
                upsampling_factor=upsampling_factor,
                val_size=job.val_size,
                test_size=job.test_size,
                only_test=False,
                llm_csv_base_path=str(
                    llm_csv_base_path,
                ),
            )
        else:
            train_class_counts = job.rebuild_config.get(
                'class_counts',
            )
            train_class_percents = job.rebuild_config.get(
                'class_percents',
            )
            train_total_samples = job.rebuild_config.get(
                'total_samples',
            )
            ds = Dataset.build_imbalanced_train_balanced_eval(
                orig_ds,
                train_class_counts=train_class_counts,
                train_class_percents=train_class_percents,
                train_total_samples=train_total_samples,
                val_size=job.val_size,
                test_size=job.test_size,
                seed=job.seed,
                only_test=False,
            )
    else:
        ds = Dataset(
            job.train_dataset,
            model_name=config['model_name'],
            device=config['device'],
            seed=job.seed,
        )
        if not job.on_policy and 'model_check' in config:
            run_name = config.get(
                'run_name',
                'default_run',
            )
            for check in config['model_check']:
                check_on = check['check_on']
                datasets_to_check = check_on if isinstance(
                    check_on,
                    list,
                ) else ([check_on] if check_on else [])
                if job.train_dataset in datasets_to_check:
                    ds.filter_data_by_model_check(
                        run_name,
                        check['name'],
                    )
                    break
        ds.split_data(
            train_size=job.train_size,
            val_size=job.val_size,
            test_size=job.test_size,
            seed=job.seed,
        )

    return ds


def run_batched_jobs(
    all_jobs: List[ProbeJob],
    config: Dict,
    results_dir: Path,
    cache_dir: Path,
    log_file_path: Path,
):
    """Process jobs in groups sharing the same data/split settings, reusing one Dataset per group."""
    # Group by data-identical keys
    groups: Dict[Tuple, List[ProbeJob]] = defaultdict(
        list,
    )
    for job in all_jobs:
        key = (
            job.train_dataset,
            job.layer,
            job.component,
            job.seed,
            _canonical_rebuild_key(
                job.rebuild_config,
            ),
            bool(
                job.on_policy,
            ),
            float(
                job.train_size,
            ),
            float(
                job.val_size,
            ),
            float(
                job.test_size,
            ),
        )
        groups[key].append(
            job,
        )

    total_groups = len(
        groups,
    )
    print(
        f"Running {sum(len(v,) for v in groups.values())} jobs in {total_groups} groups (batched)",
    )

    def _process_group(
        group_index: int,
        key: Tuple,
        jobs: List[ProbeJob],
    ) -> bool:
        print(
            f"\n=== Group {group_index}/{total_groups} — {key} ===",
        )
        # Build shared training dataset
        shared_train_ds = _build_group_dataset(
            jobs[0],
            config,
        )

        # Prepare eval datasets cache per eval dataset name (build once per group)
        eval_ds_cache: Dict[str, Dataset] = {}
        unique_eval_datasets = set()
        for j in jobs:
            for ds_name in j.eval_datasets:
                if ds_name != j.train_dataset:
                    unique_eval_datasets.add(
                        ds_name,
                    )

        if unique_eval_datasets:
            j0 = jobs[0]
            for eval_dataset in sorted(unique_eval_datasets, ):
                if j0.rebuild_config is not None:
                    orig_ds = Dataset(
                        eval_dataset,
                        model_name=config['model_name'],
                        device=config['device'],
                        seed=j0.seed,
                        only_test=True,
                    )
                    if 'llm_upsampling' in j0.rebuild_config and j0.rebuild_config['llm_upsampling']:
                        n_real_neg = j0.rebuild_config.get(
                            'n_real_neg',
                        )
                        n_real_pos = j0.rebuild_config.get(
                            'n_real_pos',
                        )
                        upsampling_factor = j0.rebuild_config.get(
                            'upsampling_factor',
                        )
                        results_path = Path(
                            config.get(
                                'results_dir',
                                '.',
                            ),
                        )
                        run_name = config.get(
                            'run_name',
                            results_path.name,
                        )
                        # Base path to results; Dataset helper will append seed and dataset
                        llm_csv_base_path = Path(
                            j0.rebuild_config.get(
                                'llm_csv_base_path',
                                results_path,
                            ),
                        )
                        eval_ds = Dataset.build_llm_upsampled_dataset(
                            orig_ds,
                            seed=j0.seed,
                            n_real_neg=n_real_neg,
                            n_real_pos=n_real_pos,
                            upsampling_factor=upsampling_factor,
                            val_size=j0.val_size,
                            test_size=j0.test_size,
                            only_test=True,
                            llm_csv_base_path=str(
                                llm_csv_base_path,
                            ),
                        )
                    else:
                        train_class_counts = j0.rebuild_config.get(
                            'class_counts',
                        )
                        train_class_percents = j0.rebuild_config.get(
                            'class_percents',
                        )
                        train_total_samples = j0.rebuild_config.get(
                            'total_samples',
                        )
                        eval_ds = Dataset.build_imbalanced_train_balanced_eval(
                            orig_ds,
                            train_class_counts=train_class_counts,
                            train_class_percents=train_class_percents,
                            train_total_samples=train_total_samples,
                            val_size=j0.val_size,
                            test_size=j0.test_size,
                            seed=j0.seed,
                            only_test=True,
                        )
                else:
                    eval_ds = Dataset(
                        eval_dataset,
                        model_name=config['model_name'],
                        device=config['device'],
                        seed=j0.seed,
                        only_test=True,
                    )
                    eval_ds.split_data(
                        train_size=j0.train_size,
                        val_size=j0.val_size,
                        test_size=j0.test_size,
                        seed=j0.seed,
                    )
                eval_ds_cache[eval_dataset] = eval_ds

        for job in jobs:
            # Create per-job directories and logger as in process_probe_job
            seed_dir = results_dir / f"seed_{job.seed}"
            experiment_dir = seed_dir / job.experiment_name
            experiment_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            trained_dir = experiment_dir / "trained"
            val_eval_dir = experiment_dir / "val_eval"
            test_eval_dir = experiment_dir / "test_eval"
            gen_eval_dir = experiment_dir / "gen_eval"
            for dir_path in [trained_dir, val_eval_dir, test_eval_dir, gen_eval_dir]:
                dir_path.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            logger = Logger(
                log_file_path,
            )
            try:
                # Train with shared dataset
                logger.log(
                    f"  🚀 [BATCH] Training probe: {job.architecture_name}",
                )
                train_single_probe(
                    job=job,
                    config=config,
                    results_dir=trained_dir,
                    cache_dir=cache_dir,
                    logger=logger,
                    rerun=rerun,
                    dataset=shared_train_ds,
                )

                # Evaluate
                logger.log(
                    f"  🤔 [BATCH] Evaluating on {len(job.eval_datasets,)} datasets…",
                )
                for eval_dataset in job.eval_datasets:
                    if eval_dataset == job.train_dataset:
                        # Val (only_test=False) then test (only_test=True) using shared train ds
                        logger.log(
                            f"    [BATCH] Evaluating on {eval_dataset} validation set",
                        )
                        try:
                            evaluate_single_probe(
                                job=job,
                                eval_dataset=eval_dataset,
                                config=config,
                                results_dir=val_eval_dir,
                                cache_dir=cache_dir,
                                logger=logger,
                                only_test=False,
                                rerun=rerun,
                                dataset=shared_train_ds,
                            )
                        except Exception as e:
                            logger.log(
                                f"    💀💀💀 ERROR evaluating validation on '{eval_dataset}': {e}",
                            )

                        logger.log(
                            f"    [BATCH] Evaluating on {eval_dataset} test set",
                        )
                        try:
                            evaluate_single_probe(
                                job=job,
                                eval_dataset=eval_dataset,
                                config=config,
                                results_dir=test_eval_dir,
                                cache_dir=cache_dir,
                                logger=logger,
                                only_test=True,
                                rerun=rerun,
                                dataset=shared_train_ds,
                            )
                        except Exception as e:
                            logger.log(
                                f"    💀💀💀 ERROR evaluating test on '{eval_dataset}': {e}",
                            )
                    else:
                        # Different dataset: reuse prebuilt eval dataset
                        try:
                            evaluate_single_probe(
                                job=job,
                                eval_dataset=eval_dataset,
                                config=config,
                                results_dir=gen_eval_dir,
                                cache_dir=cache_dir,
                                logger=logger,
                                only_test=True,
                                rerun=rerun,
                                dataset=eval_ds_cache[eval_dataset],
                            )
                        except Exception as e:
                            logger.log(
                                f"    💀💀💀 ERROR evaluating on '{eval_dataset}': {e}",
                            )
            finally:
                logger.close()

        # Proactively free memory held by datasets and caches for this group
        try:
            del shared_train_ds
            # Drop references to eval datasets
            for _ds in list(eval_ds_cache.values(), ):
                del _ds
            del eval_ds_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.log(
                f"Warning: group-level cleanup failed: {e}",
            )

        return True

    # Execute groups in parallel using threads (CPU parallelism)
    group_items = list(
        groups.items(),
    )
    # Use process-based parallelism and cap concurrency conservatively to avoid oversubscription
    # Allow overrides via env vars
    # Use a safe default of 1 job if the environment variable is not set
    # This controls the number of parallel processes (CPU parallelism)
    env_n = int(os.environ.get("EP_CPU_NJOBS", "15"))
    backend = os.environ.get(
        "EP_BATCHED_BACKEND",
        "loky",
    )
    results = []
    try:
        # Parenthesize the generator/list to satisfy Python syntax and joblib expectations
        results = Parallel(
            n_jobs=env_n,
            backend=backend,
            verbose=0,
        )([delayed(_process_group)(i + 1, key, jobs) for i, (key, jobs) in enumerate(group_items)])
    finally:
        done = sum(1 for r in results if r)
        print(f"Completed {done}/{len(group_items)} groups successfully")


def process_probe_job(
    job: ProbeJob,
    config: Dict,
    results_dir: Path,
    cache_dir: Path,
    log_file_path: Path,
):
    """Process a single probe job (train and evaluate)."""
    # Create a separate logger for this process
    logger = Logger(
        log_file_path,
    )

    logger.log(
        "-" * 80,
    )
    rebuild_str = resample_params_to_str(
        job.rebuild_config,
    ) if job.rebuild_config else "default"
    logger.log(
        f"🫠 Probe Job: {job.experiment_name}, {job.train_dataset}, L{job.layer}, {job.component}, rebuild={rebuild_str}, seed={job.seed}",
    )

    try:
        # Create seed-specific directory structure
        seed_dir = results_dir / f"seed_{job.seed}"
        experiment_dir = seed_dir / job.experiment_name
        experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Create subdirectories within experiment directory
        trained_dir = experiment_dir / "trained"
        val_eval_dir = experiment_dir / "val_eval"
        test_eval_dir = experiment_dir / "test_eval"
        gen_eval_dir = experiment_dir / "gen_eval"

        for dir_path in [trained_dir, val_eval_dir, test_eval_dir, gen_eval_dir]:
            dir_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        # Train the probe
        logger.log(
            f"  🚀 Training probe: {job.architecture_name}",
        )
        train_single_probe(
            job=job,
            config=config,
            results_dir=trained_dir,
            cache_dir=cache_dir,
            logger=logger,
            rerun=rerun,
        )

        # Evaluate on all evaluation datasets
        logger.log(
            f"  🤔 Evaluating probe on {len(job.eval_datasets,)} datasets...",
        )

        for eval_dataset in job.eval_datasets:
            # Determine which evaluation directory to use
            if eval_dataset == job.train_dataset:
                # Same dataset - evaluate on both validation and test sets
                # First evaluate on validation set
                logger.log(
                    f"    Evaluating on {eval_dataset} validation set",
                )
                try:
                    evaluate_single_probe(
                        job=job,
                        eval_dataset=eval_dataset,
                        config=config,
                        results_dir=val_eval_dir,
                        cache_dir=cache_dir,
                        logger=logger,
                        only_test=False,  # Use validation set
                        rerun=rerun,
                    )
                    logger.log(
                        f"    ✅ Validation evaluation complete for {eval_dataset}",
                    )
                except Exception as e:
                    logger.log(
                        f"    💀💀💀 ERROR evaluating validation on '{eval_dataset}': {e}",
                    )

                # Then evaluate on test set
                logger.log(
                    f"    Evaluating on {eval_dataset} test set",
                )
                eval_dir = test_eval_dir
                only_test = True
            else:
                # Different dataset - use gen_eval with only_test=True
                eval_dir = gen_eval_dir
                only_test = True

            logger.log(
                f"    Evaluating on {eval_dataset} (only_test={only_test})",
            )

            try:
                evaluate_single_probe(
                    job=job,
                    eval_dataset=eval_dataset,
                    config=config,
                    results_dir=eval_dir,
                    cache_dir=cache_dir,
                    logger=logger,
                    only_test=only_test,
                    rerun=rerun,
                )

                logger.log(
                    f"    ✅ Evaluation complete for {eval_dataset}",
                )

            except Exception as e:
                logger.log(
                    f"    💀💀💀 ERROR evaluating on '{eval_dataset}': {e}",
                )
                continue

        logger.log(
            f"  😜 Completed probe job",
        )
        return True

    except Exception as e:
        logger.log(
            f"  💀💀💀 ERROR in probe job: {e}",
        )
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
        with open(
                f"configs/{config_yaml}",
                "r",
        ) as f:
            config = yaml.safe_load(
                f,
            )
        # Re-apply device fix if needed
        device = config.get(
            "device",
        )
        if device and "cuda" in device and "1" in device:
            config["device"] = "cuda:0"
            print(
                f"Updated device from {device} to cuda:0",
            )
    except:
        print(
            f"A config of name {config_yaml} does not exist.",
        )
        return

    run_name = config.get(
        'run_name',
        'default_run',
    )
    results_dir = Path(
        "results",
    ) / run_name
    cache_dir = Path(
        "activation_cache",
    ) / config['model_name']
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Get log file path from config
    log_file_path = config.get(
        'log_file',
        results_dir / "output.log",
    )
    if isinstance(
            log_file_path,
            str,
    ):
        log_file_path = Path(
            log_file_path,
        )
        if not log_file_path.is_absolute():
            log_file_path = results_dir / log_file_path

    # Ensure the log file directory exists
    log_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = Logger(
        log_file_path,
    )

    # Get all seeds to process
    all_seeds = config.get(
        'seeds',
        [42],
    )
    if not isinstance(
            all_seeds,
            list,
    ):
        all_seeds = [all_seeds]
    logger.log(
        f"Processing {len(all_seeds,)} seeds: {all_seeds}",
    )

    # Step 1: Run model checks
    run_model_checks(
        config,
        logger,
    )

    # Step 2: Run LLM upsampling if needed
    run_llm_upsampling(
        config,
        logger,
    )

    try:
        # Create all probe jobs
        logger.log(
            "\n" + "=" * 25 + " JOB CREATION PHASE " + "=" * 25,
        )
        all_probe_jobs = create_probe_jobs(
            config,
            all_seeds,
        )
        logger.log(
            f"Created {len(all_probe_jobs,)} probe jobs",
        )

        # Process all probe jobs in batched mode
        logger.log(
            "\n" + "=" * 25 + " PROBE PROCESSING PHASE (BATCHED) " + "=" * 25,
        )
        logger.log(
            f"Processing {len(all_probe_jobs,)} probe jobs in batched groups…",
        )

        # Make results_dir available to helpers that expect it in config
        config['results_dir'] = str(
            results_dir,
        )

        run_batched_jobs(
            all_jobs=all_probe_jobs,
            config=config,
            results_dir=results_dir,
            cache_dir=cache_dir,
            log_file_path=log_file_path,
        )

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.log(
            "=" * 80,
        )
        logger.log(
            "😜 Run finished. Closing log file.",
        )
        logger.close()


if __name__ == "__main__":
    main()
