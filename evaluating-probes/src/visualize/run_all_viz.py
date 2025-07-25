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
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config name (without _config.yaml)')
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
    exp_folders = {k: None for k in ['2', '3', '4']}
    for sub in os.listdir(results_dir):
        for k in exp_folders:
            if sub.startswith(f'{k}-'):
                exp_folders[k] = results_dir / sub
    # Only keep those that exist
    exp_folders = {k: v for k, v in exp_folders.items() if v is not None}
    # For each architecture, run multi-folder recall@fpr and auc_vs_n_class1
    if exp_folders:
        dataclass_folders = []
        folder_labels = []
        for k, folder in exp_folders.items():
            # Try dataclass_exps_* first, fallback to train_*
            subfolders = [d for d in os.listdir(folder) if os.path.isdir(folder / d)]
            dc = [d for d in subfolders if d.startswith('dataclass_exps_')]
            tr = [d for d in subfolders if d.startswith('train_')]
            if dc:
                dataclass_folders.append(str(folder / dc[0]))
                folder_labels.append(f"{k}-{os.path.basename(folder)}")
            elif tr:
                dataclass_folders.append(str(folder / tr[0]))
                folder_labels.append(f"{k}-{os.path.basename(folder)}")
        colors = [f"C{i}" for i in range(len(dataclass_folders))]
        for arch in architectures:
            # Recall@FPR
            save_path = viz_root / f"recall_at_fpr_{arch}.png"
            plot_multi_folder_recall_at_fpr(
                dataclass_folders, folder_labels, arch, class_names=class_names, save_path=save_path, colors=colors
            )
            # AUC vs n_class1
            save_path = viz_root / f"auc_vs_n_class1_{arch}.png"
            plot_multi_folder_auc_vs_n_class1(
                dataclass_folders, folder_labels, arch, class_names=class_names, save_path=save_path, colors=colors
            )

    # 3. For each of the experiment folders 1-, 2-, 3-, 4-, plot all probe loss curves
    for k in ['1', '2', '3', '4']:
        for sub in os.listdir(results_dir):
            if sub.startswith(f'{k}-'):
                exp_dir = results_dir / sub
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
                plot_all_probe_loss_curves_in_folder(str(loss_folder), save_path=save_path)

if __name__ == '__main__':
    main() 