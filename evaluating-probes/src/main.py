import yaml
import itertools
from pathlib import Path
from typing import cast, Dict, Any
import sys

from src.runner import run_single_experiment
from src.data import Dataset, get_available_datasets
from src.logger import Logger
from transformer_lens import HookedTransformer

def main():
    """Main entry point for running all experiments from a config file."""
    with open("configs/main_config.yaml", "r") as f:
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

        # Expand 'all' and 'self' keywords to generate a list of concrete experiments
        all_experiments = []
        for experiment in config['experiments']:
            train_sets = [experiment['train_on']]
            if experiment['train_on'] == "all":
                train_sets = available_datasets

            for train_dataset in train_sets:
                eval_sets = experiment['evaluate_on']
                if "all" in eval_sets:
                    eval_sets = available_datasets
                if "self" in eval_sets:
                    eval_sets = [d if d != "self" else train_dataset for d in eval_sets]
                
                all_experiments.append({
                    "name": experiment.get('name', 'Unnamed') + f"_{train_dataset}",
                    "train_on": train_dataset,
                    "evaluate_on": eval_sets
                })

        for experiment in all_experiments:
            train_dataset_name = experiment['train_on']
            
            # Determine task type BEFORE creating the jobs
            train_data = Dataset(train_dataset_name, seed=global_seed)
            task_type = train_data.task_type
            
            architectures, aggregations = [], []
            if "Classification" in task_type:
                architectures = config.get('architectures', [])
                aggregations = config.get('aggregations', [])
            elif "Continuous" in task_type:
                architectures = config.get('regression_architectures', [])
                aggregations = config.get('regression_aggregations', [])
            else:
                logger.log(f"  - ⏭️  Skipping dataset '{train_dataset_name}' with unknown task type: {task_type}")
                continue

            job_list = list(itertools.product(
                experiment['evaluate_on'],
                config['layers'],
                config['components'],
                architectures,
                aggregations
            ))

            logger.log(f"--- Starting Experiment: {experiment.get('name', 'Unnamed')} ---")
            logger.log(f"  Training on: {train_dataset_name} ({task_type}), with {len(job_list)} jobs in total.")
            
            for i, (eval_dataset, layer, comp, arch_config_raw, agg) in enumerate(job_list):
                arch_config = cast(Dict[str, str], arch_config_raw)
                
                try:
                    run_single_experiment(
                        model_name=config['model_name'],
                        d_model=d_model,
                        train_dataset_name=train_dataset_name,
                        eval_dataset_name=eval_dataset,
                        layer=layer,
                        component=comp,
                        architecture_config=arch_config,
                        aggregation=agg,
                        device=config['device'],
                        use_cache=config['cache_activations'],
                        seed=global_seed,
                        results_dir=results_dir,
                        cache_dir=cache_dir,
                        logger=logger
                    )
                except Exception as e:
                    logger.log(f"❌ Job failed with error: {e}")
                    import traceback
                    traceback.print_exc(file=logger.log_file)
    finally:
        logger.log("=" * 60)
        logger.log("✅ Run finished. Closing log file.")
        logger.close()

if __name__ == "__main__":
    main()