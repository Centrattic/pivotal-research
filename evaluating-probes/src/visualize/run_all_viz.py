import argparse
import yaml
import os
from pathlib import Path
import glob
import re
from src.visualize.utils_viz import plot_logit_diffs_by_class, plot_probe_score_violins_from_folder, plot_rebuild_experiment_results_grid, plot_recall_at_fpr_from_folder, plot_auc_vs_n_class1_from_folder
import numpy as np
from src.probes import LinearProbe, AttentionProbe
from src.data import Dataset
from transformer_lens import HookedTransformer

# Add configs so you can select which plots to run with flags

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
    parser.add_argument('-e', '--experiment', required=False, help='Experiment name (from config) to visualize')
    args = parser.parse_args()
    config_path = Path('configs') / f'{args.config}_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    run_name = config.get('run_name', 'default_run')
    experiments = config.get('experiments', [])
    experiment_names = [exp['name'] for exp in experiments]
    if args.experiment:
        if args.experiment not in experiment_names:
            print(f"Experiment '{args.experiment}' not found in config. Available: {experiment_names}")
            return
        experiment = next(exp for exp in experiments if exp['name'] == args.experiment)
    else:
        experiment = experiments[0]
        print(f"No experiment specified, defaulting to: {experiment['name']}")
    experiment_name = experiment['name']
    experiment_dir = Path('results') / run_name / experiment_name
    viz_dir = experiment_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    # Load model for dataset activations
    d_model = config['d_model']
    model = None
    # 1. Model check visualizations
    if 'model_check' in config:
        for check in config['model_check']:
            ds_name = check['check_on']
            class_names = check.get('class_names', {0: 'Class0', 1: 'Class1'})
            probe_type = check.get('probe_type', 'linear')
            layer = config['layers'][0] if 'layers' in config and config['layers'] else 0
            component = config['components'][0] if 'components' in config and config['components'] else 'resid_post'
            probe_filename_base = f"{ds_name}_{probe_type}_last_L{layer}_{component}"
            probe_dir = experiment_dir / f"train_{ds_name}"
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
    # 2. Rebuild config visualizations (grid for all probes, dataclass only)
    train_on = experiment['train_on']
    dataclass_results_dir = experiment_dir / f"dataclass_exps_{train_on}"
    if dataclass_results_dir.exists():
        class_names = experiment.get('class_names', config.get('class_names', {0: 'Class0', 1: 'Class1'}))
        rebuild_configs = []
        if 'rebuild_config' in experiment:
            rc = experiment['rebuild_config']
            if isinstance(rc, dict):
                for group in rc.values():
                    rebuild_configs.extend(group)
            else:
                rebuild_configs.extend(rc)
        dataset = experiment['train_on']
        allres_files = sorted(glob.glob(str(dataclass_results_dir / '*_results.json')))
        probe_names = sorted(set(
            re.search(rf'train_on_{re.escape(dataset)}_.*?(?=_class|_results)', os.path.basename(f)).group(0)
            for f in allres_files if re.search(rf'train_on_{re.escape(dataset)}_.*?(?=_class|_results)', os.path.basename(f))
        ))
        if allres_files and rebuild_configs:
            save_path = viz_dir / f"rebuild_experiment_results_grid_dataclass_{train_on}.png"
            plot_rebuild_experiment_results_grid(
                str(dataclass_results_dir),
                probe_names,
                class_names,
                rebuild_configs,
                save_path=save_path,
                ncols=2,
                y_log_scale=False,
                show_value_labels=True
            )
    # 3. Probe score violin plots (unchanged)
    train_folder = experiment_dir / f"train_{train_on}"
    class_names = experiment.get('class_names', config.get('class_names', {0: 'Class0', 1: 'Class1'}))
    
    save_path = viz_dir / f"probe_score_violins_{train_on}.png"
    plot_probe_score_violins_from_folder(str(train_folder), class_names=class_names, save_path=str(save_path))

    # 4. Recall@FPR visualization for the new experiment
    # Fix the condition for this
    train_folder = experiment_dir / f"dataclass_exps_{experiment['train_on']}"
    class_names = experiment.get('class_names', config.get('class_names', {0: 'Class0', 1: 'Class1'}))
    save_path = viz_dir / "recall_at_fpr.png"
    plot_recall_at_fpr_from_folder(str(train_folder), class_names=class_names, save_path=str(save_path), fpr_target=0.01)
    # AUC vs n_class1 plot
    auc_save_path = viz_dir / "auc_vs_n_class1.png"
    plot_auc_vs_n_class1_from_folder(str(train_folder), class_names=class_names, save_path=str(auc_save_path))

if __name__ == '__main__':
    main() 