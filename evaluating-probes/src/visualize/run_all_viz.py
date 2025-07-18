import argparse
import yaml
import os
from pathlib import Path
import glob
from src.visualize.utils_viz import plot_logit_diffs_by_class, plot_class_logit_distributions, plot_probe_score_histogram_subplots, plot_rebuild_experiment_results_grid
import numpy as np
from src.probes import LinearProbe, AttentionProbe
from src.data import Dataset
from transformer_lens import HookedTransformer

def try_load_probe_scores_and_labels(results_dir, train_on, arch, layer, component, eval_on):
    # Placeholder: try to load from a standard file or result json
    # You may need to adapt this to your actual file structure
    probe_filename_base = f"{train_on}_{arch['name']}_{arch['aggregation']}_L{layer}_{component}"
    test_scores_path = results_dir / f"train_{train_on}" / f"test_scores_{eval_on}_{probe_filename_base}.json"
    if test_scores_path.exists():
        import json
        with open(test_scores_path, 'r') as f:
            d = json.load(f)
        return d.get('scores'), d.get('labels')
    # If not found, return None
    return None, None

def load_probe_and_test_data(probe_path, probe_type, d_model, device, dataset_name, model, layer, component, seed=42):
    """
    Loads a trained probe and test set activations/labels for visualization.
    Args:
        probe_path: Path to the saved probe .npz file
        probe_type: 'linear' or 'attention'
        d_model: model dimension
        device: device string
        dataset_name: name of the dataset
        model: HookedTransformer model (for Dataset)
        layer: layer index
        component: component name
        seed: random seed
    Returns:
        (probe, test_acts, test_labels)
    """
    # Load probe
    if probe_type == 'linear':
        probe = LinearProbe(d_model=d_model, device=device)
    elif probe_type == 'attention':
        probe = AttentionProbe(d_model=d_model, device=device)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    probe.load_state(probe_path)
    # Load test set activations and labels
    ds = Dataset(dataset_name, model=model, device=device, seed=seed)
    ds.split_data(test_size=0.15, seed=seed)
    test_acts, test_labels = ds.get_test_set_activations(layer, component)
    return probe, test_acts, test_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config name (without _config.yaml)')
    args = parser.parse_args()
    config_path = Path('configs') / f'{args.config}_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    run_name = config.get('run_name', 'default_run')
    results_dir = Path('results') / run_name
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    # Load model for dataset activations
    model = HookedTransformer.from_pretrained(config['model_name'], config['device'])
    d_model = model.cfg.d_model
    # 1. Model check visualizations
    if 'model_check' in config:
        for check in config['model_check']:
            ds_name = check['check_on']
            class_names = check.get('class_names', {0: 'Class0', 1: 'Class1'})
            probe_type = check.get('probe_type', 'linear')
            layer = config['layers'][0] if 'layers' in config and config['layers'] else 0
            component = config['components'][0] if 'components' in config and config['components'] else 'resid_post'
            # Find the probe path (assume saved in train_{ds_name} with standard naming)
            probe_filename_base = f"{ds_name}_{probe_type}_last_L{layer}_{component}"
            probe_dir = results_dir / f"train_{ds_name}"
            probe_path = probe_dir / f"train_on_{ds_name}_{probe_type}_last_L{layer}_{component}_state.npz"
            if not probe_path.exists():
                print(f"Probe file not found: {probe_path}, skipping model_check plot.")
                continue
            probe, test_acts, test_labels = load_probe_and_test_data(
                probe_path=probe_path,
                probe_type=probe_type,
                d_model=d_model,
                device=config['device'],
                dataset_name=ds_name,
                model=model,
                layer=layer,
                component=component
            )
            plot_logit_diffs_by_class(
                probe,
                test_acts,
                test_labels,
                class_names=class_names,
                save_path=str(viz_dir / f'logit_diff_hist_{ds_name}.png')
            )
    # 2. Rebuild config visualizations (grid for all probes)
    results_json_paths = []
    probe_names = []
    class_names = None
    rebuild_configs = []
    for experiment in config.get('experiments', []):
        train_on = experiment['train_on']
        for arch in config.get('architectures', []):
            for layer in config['layers']:
                for component in config['components']:
                    probe_filename_base = f"{train_on}_{arch['name']}_{arch['aggregation']}_L{layer}_{component}"
                    for eval_on in experiment['evaluate_on']:
                        results_json = results_dir / f"train_{train_on}" / f"eval_on_{eval_on}__{probe_filename_base}_allres.json"
                        if results_json.exists():
                            results_json_paths.append(str(results_json))
                            probe_names.append(f"{arch['name']}_{arch['aggregation']}_L{layer}_{component}")
                        else:
                            print(f"Results JSON not found: {results_json}")
        # Try to get class_names from experiment or config
        if not class_names:
            class_names = experiment.get('class_names', config.get('class_names', {0: 'Class0', 1: 'Class1'}))
        # Collect rebuild_configs from experiment
        if 'rebuild_config' in experiment:
            rebuild_configs.extend(experiment['rebuild_config'])
    if results_json_paths and rebuild_configs:
        save_path = viz_dir / "rebuild_experiment_results_grid.png"
        plot_rebuild_experiment_results_grid(results_json_paths, probe_names, class_names, rebuild_configs, save_path=save_path)
    # 3. Probe score histograms (unchanged)
    for experiment in config.get('experiments', []):
        train_on = experiment['train_on']
        for arch in config.get('architectures', []):
            for layer in config['layers']:
                for component in config['components']:
                    probe_filename_base = f"{train_on}_{arch['name']}_{arch['aggregation']}_L{layer}_{component}"
                    for eval_on in experiment['evaluate_on']:
                        scores, labels = try_load_probe_scores_and_labels(results_dir, train_on, arch, layer, component, eval_on)
                        if scores is not None and labels is not None:
                            probe_name = f"{arch['name']}_{arch['aggregation']}_L{layer}_{component}"
                            probe_results = {probe_name: {'scores': scores, 'labels': labels}}
                            save_path = viz_dir / f"probe_score_histograms_{eval_on}_{probe_name}.png"
                            plot_probe_score_histogram_subplots(probe_results, class_names=None, save_path=str(save_path))
                        else:
                            print(f"Test scores/labels not found for {train_on}, {arch['name']}, L{layer}, {component}, eval_on={eval_on}")

if __name__ == '__main__':
    main() 