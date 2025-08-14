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
    plot_experiment_3_per_probe,
    get_best_probes_by_type,
    plot_experiment_2_unified,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config name (without _config.yaml)')
    # Adding these as args since I don't want to save separate visualizations for filtered and all scores, can just re-run visualizations
    parser.add_argument('--filtered', dest='filtered', action='store_true', help='Use filtered scores (default)')
    parser.add_argument('--all', dest='filtered', action='store_false', help='Use all scores (not filtered)')
    parser.set_defaults(filtered=True)
    parser.add_argument('--seeds', nargs='+', default=['42'], help='List of seeds to use (default: 42)')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing visualizations (default: skip if they exist)'
    )
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
                    if not save_path.exists() or args.force:
                        plot_logit_diffs_from_csv(csv_path, class_names, save_path=save_path)
                    else:
                        print(f"Skipping {save_path} (already exists, use --force to overwrite)")

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

    # 3. Plot probe loss curves (aggregated across seeds)
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
                    if not save_path.exists() or args.force:
                        plot_all_probe_loss_curves_in_folder(str(loss_folder), save_path=save_path, seeds=args.seeds)
                    else:
                        print(f"Skipping {save_path} (already exists, use --force to overwrite)")

    # 4. Experiment-specific visualizations (aggregated across seeds)
    # Check for experiments once, outside the architecture loop
    exp2_exists = False
    exp3_exists = False
    if seed_dir.exists():
        for sub in os.listdir(seed_dir):
            if sub.startswith('2-'):
                exp2_exists = True
            elif sub.startswith('3-'):
                exp3_exists = True

    # Create experiment 2 visualizations (aggregated across seeds)
    if exp2_exists:
        print(f"Creating experiment 2 visualizations (aggregated across seeds)")

        # Get evaluation datasets from experiment 2 config - make it flexible
        eval_datasets = []
        exp2_name = None
        for exp in config.get('experiments', []):
            if exp.get('name', '').startswith('2-') and 'increasing' in exp.get('name', ''):
                eval_datasets = exp.get('evaluate_on', [])
                exp2_name = exp.get('name')
                break

        if not eval_datasets:
            print("No evaluation datasets found in experiment 2 config")
            return

        # Define probe groups - make them flexible based on config
        probe_groups = {
            'best_probes': None,  # Will be filled with best probes of each type
        }

        # Get probe names from config
        config_probes = []
        for arch in config.get('architectures', []):
            config_probes.append(arch.get('config_name', arch.get('name', '')))

        # Define probe label mapping for clearer legends - make it flexible
        probe_labels = {}
        for probe_name in config_probes:
            if 'sklearn_linear' in probe_name:
                agg_method = probe_name.split('_')[-1] if '_' in probe_name else 'mean'
                probe_labels[probe_name] = f'Linear ({agg_method})'
            elif 'act_sim' in probe_name:
                agg_method = probe_name.split('_')[-1] if '_' in probe_name else 'mean'
                probe_labels[probe_name] = f'Activation Similarity ({agg_method})'
            elif 'attention' in probe_name:
                probe_labels[probe_name] = 'Attention'
            elif 'sae' in probe_name:
                probe_labels[probe_name] = f'SAE ({probe_name})'
            else:
                probe_labels[probe_name] = probe_name

        # Define metrics
        metrics = ['auc', 'recall_at_fpr']

        # Get the training dataset from experiment 2 config
        train_dataset = None
        for exp in config.get('experiments', []):
            if exp.get('name') == exp2_name:
                train_dataset = exp.get('train_on')
                break

        # Iterate over evaluation datasets and metrics
        for eval_dataset in eval_datasets:
            print(f"Processing evaluation dataset: {eval_dataset}")

            # Check if this is out-of-distribution evaluation
            is_ood = (train_dataset is not None and eval_dataset != train_dataset)

            # Get the best probes of each type for this evaluation dataset
            best_probes = get_best_probes_by_type(
                results_dir,
                args.seeds,
                filtered=args.filtered,
                eval_dataset=eval_dataset
            )
            probe_groups['best_probes'] = list(best_probes.values()) if len(best_probes) >= 4 else None

            for metric in metrics:
                # Create plot title
                if is_ood:
                    title = "OOD (Email Spam to Text Spam)"
                else:
                    title = "Varying number of positive training examples\nwith 3000 negative examples"

                # Create filename
                filename = f'experiment_2_best_probes_comparison_{metric}_{eval_dataset}.png'
                save_path = viz_root / filename

                if not save_path.exists() or args.force:
                    print(f"Creating best_probes comparison plot for {metric} on {eval_dataset}")

                    # Create the plot
                    plot_experiment_2_unified(
                        results_dir,
                        probe_groups['best_probes'],
                        save_path=save_path,
                        metric=metric,
                        fpr_target=0.01,
                        filtered=args.filtered,
                        seeds=args.seeds,
                        plot_title=title,
                        eval_dataset=eval_dataset,
                        probe_labels=probe_labels
                    )
                else:
                    print(f"Skipping {save_path} (already exists, use --force to overwrite)")

    else:
        print(f"Experiment 2 not found")

    # Create experiment 3 visualizations for each individual probe (aggregated across seeds)
    if exp3_exists:
        save_path_base = viz_root / f'experiment_3_per_probe.png'
        if not save_path_base.exists() or args.force:
            print(f"Creating experiment 3 visualizations for each individual probe (aggregated across seeds)")

            # Get the training dataset from experiment 3 config
            train_dataset = None
            for exp in config.get('experiments', []):
                if exp.get('name') == '3-spam-pred-auc-llm-upsampling':
                    train_dataset = exp.get('train_on')
                    break

            plot_experiment_3_per_probe(
                results_dir,
                save_path_base=save_path_base,
                filtered=args.filtered,
                seeds=args.seeds,
                fpr_target=0.01,
                train_dataset=train_dataset
            )
    else:
        print(f"Experiment 3 not found")


main()
