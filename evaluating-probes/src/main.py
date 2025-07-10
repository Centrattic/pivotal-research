# src/main.py
import yaml
import itertools
from pathlib import Path
from src.runner import run_single_experiment
from src.data import get_available_datasets
from transformer_lens import HookedTransformer
from typing import cast, Dict

def main():
    """Main entry point for running all experiments from a config file."""
    with open("configs/main_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Pre-load model once to get config like d_model
    model = HookedTransformer.from_pretrained(config['model_name'], device='cpu')
    d_model = model.cfg.d_model
    del model

    # Create directories, if don't already exist
    results_dir = Path("results") / config['run_name']
    cache_dir = Path("activation_cache") / config['model_name']
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = config['datasets']
    if "all" in datasets:
        datasets = get_available_datasets()
        
    job_list = list(itertools.product(
        datasets,
        config['layers'],
        config['components'],
        config['architectures'],
        config['aggregations']
    ))

    print(f"Starting run '{config['run_name']}'. Total jobs: {len(job_list)}")

    for i, (dataset, layer, comp, arch_config_raw, agg) in enumerate(job_list):
        arch_config = cast(Dict[str, str], arch_config_raw)

        print("-" * 60)
        print(f"Running job {i+1}/{len(job_list)}:")
        print(f"  - Dataset: {dataset}, Layer: {layer}, Component: {comp}")
        print(f"  - Architecture: {arch_config['name']}, Config: {arch_config['config_name']}, Aggregation: {agg}")
        
        try:
            run_single_experiment(
                model_name=config['model_name'],
                d_model=d_model,
                dataset_name=dataset,
                layer=int(layer),
                component=comp,
                architecture_config=arch_config,
                aggregation=agg,
                device=config['device'],
                use_cache=config['cache_activations'],
                results_dir=results_dir,
                cache_dir=cache_dir
            )
        except Exception as e:
            print(f"Job failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 60)
    print("All jobs complete.")

if __name__ == "__main__":
    main()