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
    # ToDo: have to load model to get d_model, add this to config
    model = HookedTransformer.from_pretrained(config['model_name'], config['device']) 
    d_model = model.cfg.d_model
    # 1. Model check visualizations
    # if 'model_check' in config:
    #     for check in config['model_check']:
    #         ds_name = check['check_on']
    #         class_names = check.get('class_names', {0: 'Class0', 1: 'Class1'})
    #         probe_type = check.get('probe_type', 'linear')
    #         layer = config['layers'][0] if 'layers' in config and config['layers'] else 0
    #         component = config['components'][0] if 'components' in config and config['components'] else 'resid_post'
    #         # Find the probe path (assume saved in train_{ds_name} with standard naming)
    #         probe_filename_base = f"{ds_name}_{probe_type}_last_L{layer}_{component}"
    #         probe_dir = results_dir / f"train_{ds_name}"
    #         probe_path = probe_dir / f"train_on_{ds_name}_{probe_type}_last_L{layer}_{component}_state.npz"
    #         if not probe_path.exists():
    #             print(f"Probe file not found: {probe_path}, skipping model_check plot.")
    #             continue
    #         probe, test_acts, test_labels = load_probe_and_test_data(
    #             probe_path=probe_path,
    #             probe_type=probe_type,
    #             d_model=d_model,
    #             device=config['device'],
    #             dataset_name=ds_name,
    #             model=model,
    #             layer=layer,
    #             component=component
    #         )
    #         plot_logit_diffs_by_class(
    #             probe,
    #             test_acts,
    #             test_labels,
    #             class_names=class_names,
    #             save_path=str(viz_dir / f'logit_diff_hist_{ds_name}.png')
    #         )
    # 2. Rebuild config visualizations (grid for all probes, dataclass only)
    for experiment in config.get('experiments', []):
        train_on = experiment['train_on']
        dataclass_results_dir = results_dir / f"dataclass_exps_{train_on}"
        if not dataclass_results_dir.exists():
            print(f"Dataclass results dir does not exist: {dataclass_results_dir}")
            continue
        # Try to get class_names from experiment or config
        class_names = experiment.get('class_names', config.get('class_names', {0: 'Class0', 1: 'Class1'}))
        # Collect and flatten all rebuild_configs from experiment
        rebuild_configs = []
        if 'rebuild_config' in experiment:
            rc = experiment['rebuild_config']
            if isinstance(rc, dict):
                for group in rc.values():
                    rebuild_configs.extend(group)
            else:
                rebuild_configs.extend(rc)
        # Only expect 2 config types now: class_counts and class_percents
        # Probe names: extract from the files in dataclass_results_dir using train_on dataset
        import re
        dataset = experiment['train_on']
        allres_files = sorted(glob.glob(str(dataclass_results_dir / '*_results.json')))
        probe_names = sorted(set(
            re.search(rf'train_on_{re.escape(dataset)}_.*?(?=_class|_results)', os.path.basename(f)).group(0)
            for f in allres_files if re.search(rf'train_on_{re.escape(dataset)}_.*?(?=_class|_results)', os.path.basename(f))
        ))
        if allres_files and rebuild_configs:
            from src.visualize.utils_viz import plot_rebuild_experiment_results_grid
            save_path = viz_dir / f"rebuild_experiment_results_grid_dataclass_{train_on}.png"
            # Only 2 columns: class_counts and class_percents
            plot_rebuild_experiment_results_grid(
                str(dataclass_results_dir),
                probe_names,
                class_names,
                rebuild_configs,
                save_path=save_path,
                ncols=2,  # Only 2 columns now
                y_log_scale=False,
                show_value_labels=True
            )
    # 3. Probe score histograms (unchanged)
    for experiment in config.get('experiments', []):
        train_on = experiment['train_on']
        train_folder = results_dir / f"train_{train_on}"
        # Try to get class_names from experiment or config
        class_names = experiment.get('class_names', config.get('class_names', {0: 'Class0', 1: 'Class1'}))
        # New: Violin plot visualization
        from src.visualize.utils_viz import plot_probe_score_violins_from_folder
        save_path = viz_dir / f"probe_score_violins_{train_on}.png"
        plot_probe_score_violins_from_folder(str(train_folder), class_names=class_names, save_path=str(save_path))
        # Old histogram (now replaced):
        # for arch in config.get('architectures', []):
        #     for layer in config['layers']:
        #         for component in config['components']:
        #             probe_filename_base = f"{train_on}_{arch['name']}_{arch['aggregation']}_L{layer}_{component}"
        #             for eval_on in experiment['evaluate_on']:
        #                 scores, labels = try_load_probe_scores_and_labels(results_dir, train_on, arch, layer, component, eval_on)
        #                 if scores is not None and labels is not None:
        #                     probe_name = f"{arch['name']}_{arch['aggregation']}_L{layer}_{component}"
        #                     probe_results = {probe_name: {'scores': scores, 'labels': labels}}
        #                     save_path = viz_dir / f"probe_score_histograms_{eval_on}_{probe_name}.png"
        #                     plot_probe_score_histogram_subplots(probe_results, class_names=None, save_path=str(save_path))
        #                 else:
        #                     print(f"Test scores/labels not found for {train_on}, {arch['name']}, L{layer}, {component}, eval_on={eval_on}")
    # # 4. Logit weight distributions for all probes
    # for experiment in config.get('experiments', []):
    #     train_on = experiment['train_on']
    #     from src.visualize.utils_viz import plot_all_probe_logit_weight_distributions
    #     ln_f = getattr(model, 'ln_f', None)
    #     plot_all_probe_logit_weight_distributions(
    #         model=model,
    #         d_model=d_model,
    #         dataset_name=train_on,
    #         results_dir=str(results_dir),
    #         viz_dir=str(viz_dir),
    #         ln_f=ln_f if ln_f is not None else None,
    #         device=config['device']
    #     )

if __name__ == '__main__':
    main() 