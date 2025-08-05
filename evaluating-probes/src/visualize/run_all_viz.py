import argparse
import yaml
import os
from pathlib import Path
import glob
import re
from src.visualize.utils_viz import (
    plot_logit_diffs_from_csv,
    plot_multi_folder_recall_at_fpr,
    plot_multi_folder_auc_vs_n_class1,
    plot_all_probe_loss_curves_in_folder,
    plot_experiment_2_all_probes_with_eval,
    plot_experiment_2_recall_at_fpr,
    plot_experiment_3_upsampling_lineplot,
    plot_experiment_3_upsampling_lineplot_recall,
    plot_experiment_3_upsampling_lineplot_per_architecture,
    plot_experiment_3_upsampling_lineplot_grid,
    plot_experiment_2_per_seed,
    plot_experiment_3_per_seed,
    plot_experiment_2_total_with_error_bars,
    plot_experiment_2_recall_total_with_error_bars,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config name (without _config.yaml)')
    # Adding these as args since I don't want to save separate visualizations for filtered and all scores, can just re-run visualizations
    parser.add_argument('--filtered', dest='filtered', action='store_true', help='Use filtered scores (default)')
    parser.add_argument('--all', dest='filtered', action='store_false', help='Use all scores (not filtered)')
    parser.set_defaults(filtered=True)
    parser.add_argument('--seeds', nargs='+', default=['42'], help='List of seeds to use (default: 42)')
    args = parser.parse_args()
    
    config_path = Path('configs') / f'{args.config}_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    run_name = config.get('run_name', 'default_run')
    results_dir = Path('results') / run_name
    viz_root = results_dir / 'visualizations'
    viz_root.mkdir(parents=True, exist_ok=True)
    architectures = [a['name'] for a in config.get('architectures', [])]
    class_names = config.get('class_names', {0: 'Class0', 1: 'Class1'})

    # 1. Run logit diffs from CSV for any runthrough folder
    for sub in os.listdir(results_dir):
        if 'runthrough' in sub:
            runthrough_dir = results_dir / sub
            for file in os.listdir(runthrough_dir):
                if file.endswith('.csv'):
                    csv_path = runthrough_dir / file
                    save_path = viz_root / f'logit_diff_hist_{sub}.png'
                    plot_logit_diffs_from_csv(csv_path, class_names, save_path=save_path)

    # 2. Find experiment folders 2-, 3-, 4- for multi-folder plots
    # Now we need to look inside seed folders
    exp_folders = {k: None for k in ['2', '3', '4']}
    
    # Check the first seed to find experiment folders
    first_seed = args.seeds[0]
    seed_dir = results_dir / f"seed_{first_seed}"
    if seed_dir.exists():
        for sub in os.listdir(seed_dir):
            for k in exp_folders:
                if sub.startswith(f'{k}-'):
                    exp_folders[k] = sub  # Store just the folder name
                    break
    
    # Only keep those that exist
    exp_folders = {k: v for k, v in exp_folders.items() if v is not None}
    
    # For each architecture, run multi-folder recall@fpr and auc_vs_n_class1
    if exp_folders:
        dataclass_folders = []
        folder_labels = []
        for k, folder_name in exp_folders.items():
            # Check if dataclass_exps_* exists in the first seed
            folder_path = seed_dir / folder_name
            subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(folder_path / d)]
            dc = [d for d in subfolders if d.startswith('dataclass_exps_')]
            tr = [d for d in subfolders if d.startswith('train_')]
            if dc:
                dataclass_folders.append(str(folder_path / dc[0]))
                folder_labels.append(f"{k}-{folder_name}")
            elif tr:
                dataclass_folders.append(str(folder_path / tr[0]))
                folder_labels.append(f"{k}-{folder_name}")
        
        colors = [f"C{i}" for i in range(len(dataclass_folders))]
        # Note: Individual architecture plots removed as requested

    # 3. For each of the experiment folders 1-, 2-, 3-, 4-, plot all probe loss curves
    for k in ['1', '2', '3', '4']:
        # Check in the first seed directory
        if seed_dir.exists():
            for sub in os.listdir(seed_dir):
                if sub.startswith(f'{k}-'):
                    exp_dir = seed_dir / sub
                    # Try dataclass_exps_* first, fallback to train_*
                    subfolders = [d for d in os.listdir(exp_dir) if os.path.isdir(exp_dir / d)]
                    dc = [d for d in subfolders if d.startswith('dataclass_exps_')]
                    tr = [d for d in subfolders if d.startswith('train_')]
                    if dc:
                        loss_folder = exp_dir / dc[0]
                    elif tr:
                        loss_folder = exp_dir / tr[0]
                    else:
                        continue
                    save_path = viz_root / f'loss_curves_{sub}.png'
                    plot_all_probe_loss_curves_in_folder(str(loss_folder), save_path=save_path, seeds=args.seeds)

    # 4. New experiment-specific visualizations
    # Check for experiments once, outside the architecture loop
    exp2_exists = False
    exp3_exists = False
    if seed_dir.exists():
        for sub in os.listdir(seed_dir):
            if sub.startswith('2-'):
                exp2_exists = True
            elif sub.startswith('3-'):
                exp3_exists = True
    
    # Create experiment 2 visualizations for each seed
    if exp2_exists:
        print(f"Creating experiment 2 visualizations for each seed")
        save_path = viz_root / f'experiment_2_per_seed.png'
        plot_experiment_2_per_seed(
            results_dir, architectures, save_path=save_path, filtered=args.filtered, 
            seeds=args.seeds, config_name=args.config
        )
        
        # Create total experiment 2 plots with error bars
        print(f"Creating experiment 2 total plots with error bars")
        save_path = viz_root / f'experiment_2_total_auc.png'
        plot_experiment_2_total_with_error_bars(
            results_dir, architectures, save_path=save_path, filtered=args.filtered, 
            seeds=args.seeds, config_name=args.config
        )
        
        save_path = viz_root / f'experiment_2_total_recall.png'
        plot_experiment_2_recall_total_with_error_bars(
            results_dir, architectures, save_path=save_path, filtered=args.filtered, 
            seeds=args.seeds, config_name=args.config
        )
    else:
        print(f"Experiment 2 not found")
    
    # Create experiment 3 visualizations for each seed
    if exp3_exists:
        print(f"Creating experiment 3 visualizations for each seed")
        save_path = viz_root / f'experiment_3_per_seed.png'
        plot_experiment_3_per_seed(
            results_dir, architectures, save_path=save_path, filtered=args.filtered, 
            seeds=args.seeds, config_name=args.config
        )
    else:
        print(f"Experiment 3 not found")

main() 