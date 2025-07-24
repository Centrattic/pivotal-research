import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from src.runner import train_probe, evaluate_probe
from src.data import Dataset, get_available_datasets, load_combined_classification_datasets
from src.logger import Logger
from src.utils import should_skip_dataset
from transformer_lens import HookedTransformer

def get_dataset(name, seed):
    if name == "single_all":
        return load_combined_classification_datasets(seed)
    else:
        return Dataset(name, seed=seed)

def main():
    with open("configs/french_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_name = config.get('run_name', 'default_run')
    results_dir = Path("results") / run_name
    cache_dir = Path("activation_cache") / config['model_name']
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(results_dir / "output.log")

    try:
        logger.log(f"Loading model '{config['model_name']}' to get config...")
        model = HookedTransformer.from_pretrained(config['model_name'], device='cpu')
        d_model = model.cfg.d_model
        del model

        global_seed = int(config.get('seed', 42))
        available_datasets = get_available_datasets()

        # --- Step 1: Pre-flight check and gather all unique training jobs ---
        logger.log("\n--- Performing Pre-flight Checks and Gathering Jobs ---")
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
                            training_jobs.add((train_dataset, layer, component, arch_config['name'], arch_config['config_name']))

        # Check all datasets that will be used for either training or evaluation
        valid_dataset_metadata = {}
        for dataset_name in sorted(list(all_dataset_names_to_check)):
            try:
                data = get_dataset(dataset_name, global_seed)
                if should_skip_dataset(dataset_name, data, logger):
                    continue
                valid_dataset_metadata[dataset_name] = {
                    'task_type': getattr(data, 'task_type', None), 
                    'n_classes': getattr(data, 'n_classes', None)
                }
            except Exception as e:
                logger.log(f"  - ❌ ERROR checking dataset '{dataset_name}': {e}")

        valid_training_jobs = [job for job in training_jobs if job[0] in valid_dataset_metadata]

        # --- Step 2: Training Phase ---
        logger.log("\n" + "="*25 + " TRAINING PHASE " + "="*25)
        for i, (train_ds, layer, comp, arch_name, conf_name) in enumerate(valid_training_jobs):
            logger.log("-" * 60)
            logger.log(f"🚀 Training job {i+1}/{len(valid_training_jobs)}: {train_ds}, {arch_name}, L{layer}, {comp}")
            train_probe(
                model_name=config['model_name'], d_model=d_model, train_dataset_name=train_ds,
                layer=layer, component=comp, architecture_name=arch_name, config_name=conf_name,
                device=config['device'], use_cache=config['cache_activations'], seed=global_seed,
                results_dir=results_dir, cache_dir=cache_dir, logger=logger
            )

        # --- Step 3: Evaluation Phase ---
        logger.log("\n" + "="*25 + " EVALUATION PHASE " + "="*25)
        for experiment in config['experiments']:
            train_sets = [experiment['train_on']]
            if experiment['train_on'] == "all": train_sets = available_datasets

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
                        logger.log(f"  - ⏭️  Skipping evaluation of probe from '{train_dataset}' on '{eval_dataset}' due to task mismatch.")
                        continue

                    for arch_config in config.get('architectures', []):
                        for layer in config['layers']:
                            for component in config['components']:
                                for agg in config.get('aggregations', []):
                                    evaluate_probe(
                                        train_dataset_name=train_dataset, eval_dataset_name=eval_dataset,
                                        layer=layer, component=component, architecture_config=arch_config,
                                        aggregation=agg, results_dir=results_dir, logger=logger, seed=global_seed,
                                        model_name=config['model_name'], d_model=d_model, device=config['device'],
                                        use_cache=config['cache_activations'], cache_dir=cache_dir
                                    )
    finally:
        logger.log("=" * 60)
        logger.log("✅ Run finished. Closing log file.")
        logger.close()

if __name__ == "__main__":
    main()
