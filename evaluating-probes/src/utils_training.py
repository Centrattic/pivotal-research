from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import copy
import json
import pandas as pd
from src.data import Dataset, get_model_d_model
from src.logger import Logger
from configs.probes import PROBE_CONFIGS
from src.utils import (
    extract_aggregation_from_config,
    get_probe_architecture,
    get_probe_filename_prefix,
    rebuild_suffix,
)
from configs.probes import (
    ProbeJob,
    SAEProbeConfig,
    PytorchAttentionProbeConfig,
    MassMeanProbeConfig,
    ActivationSimilarityProbeConfig,
)


def add_hyperparams_to_filename(base_filename: str, probe_config) -> str:
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
            elif hparam == 'top_k_features':
                hparam_suffix += f"_topk{value}"

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


def extract_activations_for_dataset(
    model,
    tokenizer,
    model_name,
    dataset_name,
    layer,
    component,
    device,
    seed,
    logger,
    results_dir,
    format_type,
    on_policy,
    include_llm_samples=True,
):
    """
    Extract activations for a dataset using the full model.
    This is phase 1: ensure all activations are cached.
    Also pre-compute aggregated activations for faster loading later.
    If include_llm_samples=True, also extract activations for any LLM upsampled samples found.
    """
    logger.log(f"  - Extracting activations for {dataset_name}, L{layer}, {component} (on_policy={on_policy})")

    try:
        # Create dataset with full model to extract activations
        ds = Dataset(
            dataset_name,
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            seed=seed,
        )

        if on_policy:
            # For on-policy: extract activations using the specified format
            logger.log(f"    - On-policy: extracting activations using format '{format_type}'...")
            if hasattr(ds, 'question_texts') and ds.question_texts is not None:
                dataset_texts = ds.X.tolist()
                question_texts = ds.question_texts.tolist()
                _, newly_added = ds.act_manager.get_activations_for_texts(
                    dataset_texts,
                    layer,
                    component,
                    format_type,
                    activation_type="full",
                    question_texts=question_texts,
                    return_newly_added_count=True,
                )
                logger.log(f"    - Cached {newly_added} new activations (on-policy format: {format_type})")
            else:
                logger.log(f"    - Warning: No response texts found for on-policy dataset")
                dataset_texts = ds.X.tolist()
                _, newly_added = ds.act_manager.get_activations_for_texts(
                    dataset_texts,
                    layer,
                    component,
                    format_type,
                    activation_type="full",
                    return_newly_added_count=True,
                )
                logger.log(f"    - Cached {newly_added} new activations (dataset only)")
        else:
            # For off-policy: extract activations from prompts only
            logger.log(f"    - Off-policy: extracting activations from prompts...")
            dataset_texts = ds.X.tolist()
            _, newly_added = ds.act_manager.get_activations_for_texts(
                dataset_texts,
                layer,
                component,
                format_type,
                activation_type="full",
                return_newly_added_count=True,
            )
            logger.log(f"    - Cached {newly_added} new activations (dataset)")

        # If including LLM samples, scan results/<run_name>/seed_*/llm_samples_<dataset_name>/samples_*.csv for prompts and cache them
        llm_samples_added = 0
        if include_llm_samples and results_dir is not None:
            try:
                llm_texts = []
                # Look for dataset-specific llm_samples in the results directory structure
                for seed_dir in results_dir.glob('seed_*'):
                    llm_dir = seed_dir / f'llm_samples_{dataset_name}'
                    if not llm_dir.exists():  # No LLM samples for this dataset
                        continue
                    for csv_file in llm_dir.glob('samples_*.csv'):
                        try:
                            df = pd.read_csv(csv_file)
                            if 'prompt' in df.columns:
                                llm_texts.extend(df['prompt'].astype(str).tolist())
                        except Exception:
                            continue
                if llm_texts:
                    # Automatically recomputes aggregations too
                    _, llm_samples_added = ds.act_manager.get_activations_for_texts(
                        llm_texts,
                        layer,
                        component,
                        format_type,
                        activation_type="full",
                        return_newly_added_count=True,
                    )
                    logger.log(f"    - Cached {llm_samples_added} new activations (LLM)")
                    # Ensure aggregations are up-to-date for any new LLM activations just added
            except Exception as e:
                logger.log(f"    - ⚠️ LLM sample caching skipped due to error: {e}")

        # Aggregated activations are computed after any new full activations (dataset + LLM)
        logger.log(f"    - Aggregated activations are up-to-date")

    except Exception as e:
        logger.log(f"    - 💀💀💀 ERROR extracting activations for {dataset_name}: {e}")
        raise


def train_single_probe(
    job: ProbeJob,
    config: Dict,
    results_dir: Path,
    cache_dir: Path,
    logger: Logger,
    rerun: bool,
    dataset: Dataset | None = None,
):
    """
    Train a single probe with the given configuration.
    """

    # Create config name robustly: if the architecture name is already a known config, use it directly
    if job.architecture_name in PROBE_CONFIGS:
        config_name = job.architecture_name
    elif hasattr(job.probe_config, 'aggregation'):
        config_name = f"{job.architecture_name}_{job.probe_config.aggregation}"
    else:
        config_name = job.architecture_name

    probe_filename_base = get_probe_filename_prefix(
        job.train_dataset,
        job.architecture_name,
        job.layer,
        job.component,
        config_name,
    )
    # Save all probes directly under the trained/ directory
    probe_save_dir = results_dir

    # Add hyperparameters to filename if they differ from defaults
    probe_filename_with_hparams = add_hyperparams_to_filename(
        probe_filename_base,
        job.probe_config,
    )

    # If rebuilding, save in dataclass_exps_{dataset_name}
    if job.rebuild_config is not None:
        suffix = rebuild_suffix(job.rebuild_config)
        probe_filename = f"{probe_filename_with_hparams}_{suffix}_state.npz"
        probe_state_path = probe_save_dir / probe_filename
        probe_json_path = probe_save_dir / f"{probe_filename_with_hparams}_{suffix}_meta.json"
    else:
        probe_state_path = probe_save_dir / f"{probe_filename_with_hparams}_state.npz"
        probe_json_path = probe_save_dir / f"{probe_filename_with_hparams}_meta.json"

    # EARLY CHECK: If probe already exists and we're not rerunning, skip immediately
    if config.get('cache_activations', True) and probe_state_path.exists() and not rerun:
        logger.log(f"  - [SKIP] Probe already trained: {probe_state_path.name}")
        return

    logger.log("  - Training new probe …")

    # Prepare dataset (allow passing a prebuilt dataset for batching)
    logger.log(f"  [DEBUG] Starting dataset preparation...")
    if dataset is not None:
        train_ds = dataset
    elif job.rebuild_config is not None:
        logger.log(f"  [DEBUG] Using rebuild_config: {job.rebuild_config}")
        orig_ds = Dataset(
            job.train_dataset,
            model_name=config['model_name'],
            device=config['device'],
            seed=job.seed,
        )

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
                    orig_ds.filter_data_by_model_check(run_name, check['name'])
                    break

        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in job.rebuild_config and job.rebuild_config['llm_upsampling']:
            logger.log(f"  [DEBUG] Using LLM upsampling method")
            n_real_neg = job.rebuild_config.get('n_real_neg')
            n_real_pos = job.rebuild_config.get('n_real_pos')
            upsampling_factor = job.rebuild_config.get('upsampling_factor')

            # Determine run_name robustly from results dir: results/<run_name>/seed_<seed>/<experiment>/trained
            results_path = Path(results_dir)
            # parents[2] -> <run_name>
            run_name = results_path.parents[2].name if len(results_path.parents) >= 3 else str(results_path)
            llm_csv_base_path = job.rebuild_config.get('llm_csv_base_path', f'results/{run_name}')

            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError(
                    "For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified"
                )

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
        else:
            # Original rebuild_config logic
            logger.log(f"  [DEBUG] Using original rebuild_config logic")
            train_class_counts = job.rebuild_config.get('class_counts')
            train_class_percents = job.rebuild_config.get('class_percents')
            train_total_samples = job.rebuild_config.get('total_samples')
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
    else:
        logger.log(f"  [DEBUG] Using simple dataset creation")
        train_ds = Dataset(job.train_dataset, model_name=config['model_name'], device=config['device'], seed=job.seed)

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
                    train_ds.filter_data_by_model_check(run_name, check['name'])
                    break

        train_ds.split_data(train_size=job.train_size, val_size=job.val_size, test_size=job.test_size, seed=job.seed)

    # Get activations
    activation_type = get_activation_type_from_config(config_name)
    logger.log(f"  [DEBUG] Using activation_type: {activation_type}")

    # Get activation extraction format from config
    format_type = config['activation_extraction']['format_type']

    # Get activations using the unified method
    train_result = train_ds.get_train_set_activations(
        job.layer,
        job.component,
        format_type,
        activation_type=activation_type,
        on_policy=job.on_policy,
    )

    # Handle different return formats (with/without masks)
    if len(train_result) == 3:
        train_acts, train_masks, y_train = train_result
    else:
        train_acts, y_train = train_result
        train_masks = None

    print(f"Train activations: {len(train_acts) if isinstance(train_acts, list) else train_acts.shape}")

    # Debug: verify alignment and class distribution before training
    try:
        num_samples_X = len(train_acts) if isinstance(train_acts, list) else (
            train_acts.shape[0] if hasattr(train_acts, 'shape') else None
        )
        num_samples_y = len(y_train) if y_train is not None else None
        print(f"[DEBUG] Pre-fit sample counts — X: {num_samples_X}, y: {num_samples_y}")
        if num_samples_X is not None and num_samples_y is not None and num_samples_X != num_samples_y:
            print(f"[DEBUG] MISMATCH detected: X has {num_samples_X} samples while y has {num_samples_y}")
        # Class distribution in y_train
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique_classes.tolist(), class_counts.tolist())}
        print(f"[DEBUG] y_train class distribution: {class_dist}")
    except Exception as e:
        print(f"[DEBUG] Failed to compute pre-fit debug stats: {e}")

    # Create probe based on architecture_name
    logger.log(f"  [DEBUG] Creating probe for architecture: {job.architecture_name}")

    # Convert probe config to dict
    probe_config_dict = asdict(job.probe_config)

    if job.architecture_name == "sklearn_linear":
        logger.log(f"  [DEBUG] Creating sklearn linear probe")
        # Sklearn linear probe
        probe = get_probe_architecture(
            "sklearn_linear",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )
        logger.log(f"  [DEBUG] Created sklearn linear probe")

        logger.log(f"  [DEBUG] Fitting probe normally...")
        # Pass masks only if they exist (for pre-aggregated activations, masks=None)
        if train_masks is not None:
            probe.fit(train_acts, y_train, train_masks)
        else:
            probe.fit(train_acts, y_train)
        logger.log(f"  [DEBUG] Fitted probe normally")

    elif job.architecture_name == "attention":
        logger.log(f"  [DEBUG] Creating attention probe")
        # Attention probe
        probe = get_probe_architecture(
            "attention",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )
        logger.log(f"  [DEBUG] Fitting attention probe normally...")
        probe.fit(train_acts, y_train)

    elif job.architecture_name == "sae" or job.architecture_name.startswith("sae"):
        # SAE probe
        probe = get_probe_architecture(
            "sae",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict,
        )

        logger.log(f"  [DEBUG] Fitting SAE probe normally...")
        probe.fit(train_acts, y_train, train_masks)

    elif job.architecture_name == "mass_mean":
        # Mass mean probe (non-trainable)
        probe = get_probe_architecture(
            "mass_mean",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )
        logger.log(f"  [DEBUG] Fitting mass mean probe...")
        probe.fit(train_acts, y_train)

    elif job.architecture_name == "act_sim":
        # Activation similarity probe (non-trainable)
        probe = get_probe_architecture(
            "act_sim",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )
        logger.log(f"  [DEBUG] Fitting activation similarity probe...")
        probe.fit(train_acts, y_train)

    else:
        raise ValueError(f"Unknown architecture_name: {job.architecture_name}")

    # Save probe
    probe_save_dir.mkdir(parents=True, exist_ok=True)
    probe.save_state(probe_state_path)

    # Save metadata/config
    # yapf: disable
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
    # yapf: enable
    with open(probe_json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.log(f"  - 🔥 Probe state saved to {probe_state_path.name}")


def evaluate_single_probe(
    job: ProbeJob,
    eval_dataset: str,
    config: Dict,
    results_dir: Path,
    cache_dir: Path,
    logger: Logger,
    only_test: bool,
    rerun: bool,
    dataset: Dataset | None = None,
) -> None:
    """
    Evaluate a single probe on a given dataset.
    """
    # Create config name robustly: if the architecture name is already a known config, use it directly
    if job.architecture_name in PROBE_CONFIGS:
        config_name = job.architecture_name
    elif hasattr(job.probe_config, 'aggregation'):
        config_name = f"{job.architecture_name}_{job.probe_config.aggregation}"
    else:
        config_name = job.architecture_name

    # Extract aggregation from config for results, to be backward compatible
    agg_name = extract_aggregation_from_config(
        config_name,
        job.architecture_name,
    )

    probe_filename_base = get_probe_filename_prefix(
        job.train_dataset,
        job.architecture_name,
        job.layer,
        job.component,
        config_name,
    )
    # Add hyperparameters to filename if they differ from defaults
    probe_filename_with_hparams = add_hyperparams_to_filename(
        probe_filename_base,
        job.probe_config,
    )

    # The results_dir is already the specific evaluation directory (val_eval, test_eval, or gen_eval)
    # Probes are saved directly under the trained directory (no subfolders)
    # results/.../seed_<seed>/<experiment>/trained/
    experiment_dir = results_dir.parent  # e.g., .../seed_42/<experiment>
    trained_dir = experiment_dir / "trained"

    if job.rebuild_config is not None:
        suffix = rebuild_suffix(job.rebuild_config)
        # Rebuild states are saved directly under trained/
        probe_state_path = trained_dir / f"{probe_filename_with_hparams}_{suffix}_state.npz"
        eval_results_path = results_dir / f"eval_on_{eval_dataset}__{probe_filename_with_hparams}_{suffix}_seed{job.seed}_{agg_name}_results.json"
    else:
        # Normal training states are saved directly under trained/
        probe_state_path = trained_dir / f"{probe_filename_with_hparams}_state.npz"
        eval_results_path = results_dir / f"eval_on_{eval_dataset}__{probe_filename_with_hparams}_seed{job.seed}_{agg_name}_results.json"

    if config.get('cache_activations', True) and eval_results_path.exists() and not rerun:
        logger.log("  - 😋 Using cached evaluation result ")
        with open(eval_results_path, "r") as f:
            return json.load(f)["metrics"]

    # Load probe
    probe_config_dict = asdict(job.probe_config)

    if job.architecture_name == "sklearn_linear":
        probe = get_probe_architecture(
            "sklearn_linear",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )

    elif job.architecture_name == "attention":
        probe = get_probe_architecture(
            "attention",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )
    elif job.architecture_name == "sae" or job.architecture_name.startswith("sae"):
        probe = get_probe_architecture(
            "sae",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict,
        )
    elif job.architecture_name == "mass_mean":
        probe = get_probe_architecture(
            "mass_mean",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )
    elif job.architecture_name == "act_sim":
        probe = get_probe_architecture(
            "act_sim",
            d_model=get_model_d_model(config['model_name']),
            device=config['device'],
            config=probe_config_dict
        )
    else:
        raise ValueError(f"Unknown architecture_name: {job.architecture_name}")

    probe.load_state(probe_state_path)

    # Get batch_size from probe config for evaluation
    # For SAE probes, use training_batch_size; for others, use batch_size
    if job.architecture_name == "sae":
        batch_size = probe_config_dict.get(
            'training_batch_size',
            probe_config_dict.get('batch_size', 200),
        )
    else:
        batch_size = probe_config_dict.get('batch_size', 200)

    # Prepare evaluation dataset (allow passing a prebuilt dataset for batching)
    if dataset is not None:
        eval_ds = dataset
    elif job.rebuild_config is not None:
        orig_ds = Dataset(
            eval_dataset, model_name=config['model_name'], device=config['device'], seed=job.seed, only_test=only_test
        )

        # Check if this is LLM upsampling with new method
        if 'llm_upsampling' in job.rebuild_config and job.rebuild_config['llm_upsampling']:
            n_real_neg = job.rebuild_config.get('n_real_neg')
            n_real_pos = job.rebuild_config.get('n_real_pos')
            upsampling_factor = job.rebuild_config.get('upsampling_factor')

            # Determine run_name robustly from results dir: results/<run_name>/seed_<seed>/<experiment>/trained
            results_path = Path(results_dir)
            run_name = results_path.parents[2].name if len(results_path.parents) >= 3 else str(results_path)
            llm_csv_base_path = job.rebuild_config.get('llm_csv_base_path', f'results/{run_name}')

            if n_real_neg is None or n_real_pos is None or upsampling_factor is None:
                raise ValueError(
                    "For LLM upsampling, 'n_real_neg', 'n_real_pos', and 'upsampling_factor' must be specified"
                )

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
        eval_ds = Dataset(
            eval_dataset,
            model_name=config['model_name'],
            device=config['device'],
            seed=job.seed,
            only_test=only_test,
        )
        eval_ds.split_data(
            train_size=job.train_size,
            val_size=job.val_size,
            test_size=job.test_size,
            seed=job.seed,
        )

    # Get activations
    activation_type = get_activation_type_from_config(config_name)

    # Get activation extraction format from config
    format_type = config['activation_extraction']['format_type']

    # Get activations using the unified method
    test_result = eval_ds.get_test_set_activations(
        job.layer,
        job.component,
        format_type,
        activation_type=activation_type,
        on_policy=job.on_policy,
    )

    # Handle different return formats (with/without masks)
    if len(test_result) == 3:
        test_acts, test_masks, y_test = test_result
    else:
        test_acts, y_test = test_result
        test_masks = None

    print(f"Test activations: {len(test_acts) if isinstance(test_acts, list) else test_acts.shape}")

    # Calculate metrics
    logger.log(f"  - 🤗 Calculating metrics...")
    if job.architecture_name == "attention" or job.architecture_name == "act_sim":
        # Attention probes and activation similarity probes don't use masks
        metrics = probe.score(test_acts, y_test)
    else:
        # Other probes use masks
        metrics = probe.score(test_acts, y_test, masks=test_masks)

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

    # Build output structure
    output_dict = {"metrics": metrics, "scores": {"scores": test_scores, "labels": test_labels}}

    # Ensure the directory exists
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_results_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    # Log the results
    logger.log(f"  - ❤️‍🔥 Success! Metrics: {metrics}")
