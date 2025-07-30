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
        train_class_counts = rebuild_config.get('class_counts')
        train_class_percents = rebuild_config.get('class_percents')
        train_total_samples = rebuild_config.get('total_samples')
        llm_upsample = rebuild_config.get('llm_upsample', False)
        llm_csv_path = None
        if llm_upsample:
            run_name = str(results_dir).split('/')[-2] if 'results' in str(results_dir) else 'default_run'
            llm_csv_path = Path('results') / run_name / 'llm_samples.csv'
        train_ds = Dataset.build_imbalanced_train_balanced_eval(
            orig_ds,
            train_class_counts=train_class_counts,
            train_class_percents=train_class_percents,
            train_total_samples=train_total_samples,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            llm_upsample=llm_upsample,
            llm_csv_path=llm_csv_path
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
    
    # Handle mass-mean probe configuration specially
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
        elif weighting_method == "pcngd":
            probe.fit_pcngd(train_acts, y_train, mask=None, **fit_params)
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
        elif weighting_method == "pcngd":
            probe.fit_pcngd(train_acts, y_train, mask=None, **fit_params)
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
        eval_results_path = probe_save_dir / f"eval_on_{eval_dataset_name}__{probe_filename_base}_{agg_name}_results.json"
    else:
        probe_save_dir = results_dir / f"train_{train_dataset_name}"
        probe_state_path = probe_save_dir / f"{probe_filename_base}_state.npz"
        eval_results_path = probe_save_dir / f"eval_on_{eval_dataset_name}__{probe_filename_base}_{agg_name}_results.json"

    if use_cache and eval_results_path.exists() and not reevaluate:
        logger.log("  - 😋 Using cached evaluation result ")
        if return_metrics:
            with open(eval_results_path, "r") as f:
                return json.load(f)["metrics"]
        return

    # Load probe
    probe = get_probe_architecture(architecture_name, d_model=d_model, device=device, config=asdict(PROBE_CONFIGS[config_name]))
    probe.load_state(probe_state_path)

    if rebuild_config is not None:
        orig_ds = Dataset(eval_dataset_name, model=model, device=device, seed=seed)
        train_class_counts = rebuild_config.get('class_counts')
        train_class_percents = rebuild_config.get('class_percents')
        train_total_samples = rebuild_config.get('total_samples')
        llm_upsample = rebuild_config.get('llm_upsample', False)
        llm_csv_path = None
        if llm_upsample:
            run_name = str(results_dir).split('/')[-2] if 'results' in str(results_dir) else 'default_run'
            llm_csv_path = Path('results') / run_name / 'llm_samples.csv'
        eval_ds = Dataset.build_imbalanced_train_balanced_eval(
            orig_ds,
            train_class_counts=train_class_counts,
            train_class_percents=train_class_percents,
            train_total_samples=train_total_samples,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            llm_upsample=llm_upsample,
            llm_csv_path=llm_csv_path
        )
    else:
        eval_ds = Dataset(eval_dataset_name, model=model, device=device, seed=seed)
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
        all_metrics = probe.score(test_acts, y_test)
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
    test_scores = probe.predict_logits(test_acts)
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

