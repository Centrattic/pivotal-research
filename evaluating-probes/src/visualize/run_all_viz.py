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
    plot_experiment_unified,
    plot_probe_group_comparison,
    get_best_default_probes_by_type,
    default_probe_patterns,
    plot_scaling_law_across_runs,
)
from src.visualize import analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Config name (without _config.yaml)',
    )
    # Scoring is already filtered in the new pipeline; no flag needed
    parser.add_argument(
        '--seeds',
        nargs='+',
        default=['42'],
        help='List of seeds to use (default: 42)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing visualizations (default: skip if they exist)',
    )
    args = parser.parse_args()

    config_path = Path('configs') / f'{args.config}_config.yaml'
    with open(
            config_path,
            'r',
    ) as f:
        config = yaml.safe_load(f)
    run_name = config.get(
        'run_name',
        'default_run',
    )
    results_dir = Path('results') / run_name
    viz_root = results_dir / 'visualizations'
    viz_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    architectures = [a['name'] for a in config.get(
        'architectures',
        [],
    )]
    class_names = config.get(
        'class_names',
        {0: 'Class0', 1: 'Class1'},
    )

    # 1. Run logit diffs from CSV for any runthrough folder
    for sub in os.listdir(results_dir):
        if 'runthrough' in sub:
            runthrough_dir = results_dir / sub
            for file in os.listdir(runthrough_dir):
                if file.endswith('.csv'):
                    csv_path = runthrough_dir / file
                    save_path = viz_root / f'logit_diff_hist_{sub}.png'
                    if not save_path.exists() or args.force:
                        plot_logit_diffs_from_csv(
                            csv_path,
                            class_names,
                            save_path=save_path,
                        )
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
                        plot_all_probe_loss_curves_in_folder(
                            str(loss_folder),
                            save_path=save_path,
                            seeds=args.seeds,
                        )
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
        for exp in config.get(
                'experiments',
            [],
        ):
            if exp.get(
                    'name',
                    '',
            ).startswith('2-') and 'increasing' in exp.get(
                    'name',
                    '',
            ):
                eval_datasets = exp.get(
                    'evaluate_on',
                    [],
                )
                exp2_name = exp.get('name')
                break

        if not eval_datasets:
            print("No evaluation datasets found in experiment 2 config")
            return

        # Define probe groups
        patterns = default_probe_patterns()

        # Get probe names from config
        config_probes = []
        for arch in config.get(
                'architectures',
            [],
        ):
            config_probes.append(arch.get(
                'config_name',
                arch.get(
                    'name',
                    '',
                ),
            ))

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
        for exp in config.get(
                'experiments',
            [],
        ):
            if exp.get('name') == exp2_name:
                train_dataset = exp.get('train_on')
                break

        # Iterate over evaluation datasets and metrics
        for eval_dataset in eval_datasets:
            # Skip train dataset here; train set plots are covered by other plots
            if train_dataset is not None and eval_dataset == train_dataset:
                continue
            print(f"Processing evaluation dataset: {eval_dataset}")

            # Check if this is out-of-distribution evaluation
            is_ood = (train_dataset is not None and eval_dataset != train_dataset)

            # Get the best probes of each type for this evaluation dataset
            best_defaults = get_best_default_probes_by_type(
                results_dir,
                args.seeds,
                exp_prefix='2-',
                
                eval_dataset=eval_dataset,
            )
            best_list = list(best_defaults.values()) if best_defaults else []

            for metric in metrics:
                # Create plot title
                if is_ood:
                    title = "OOD (Email Spam to Text Spam)"
                else:
                    title = "Varying number of positive training examples\nwith 3000 negative examples"

                # Create filename
                filename = f'experiment_2_best_defaults_comparison_{metric}_{eval_dataset}.png'
                save_path = viz_root / filename

                if not save_path.exists() or args.force:
                    print(f"Creating best_probes comparison plot for {metric} on {eval_dataset}")

                    if best_list:
                        plot_experiment_unified(
                            results_dir,
                            best_list,
                            save_path=save_path,
                            metric=metric,
                            fpr_target=0.01,
                            
                            seeds=args.seeds,
                            plot_title=title,
                            eval_dataset=eval_dataset,
                            exp_prefix='2-',
                            probe_labels=probe_labels,
                            x_label=config.get('plot_labels', {}).get('x_label', None),
                            y_label=config.get('plot_labels', {}).get('y_label', None),
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
            for exp in config.get(
                    'experiments',
                [],
            ):
                if exp.get('name') == '3-spam-pred-auc-llm-upsampling':
                    train_dataset = exp.get('train_on')
                    break

            plot_experiment_3_per_probe(
                results_dir,
                save_path_base=save_path_base,
                
                seeds=args.seeds,
                fpr_target=0.01,
                train_dataset=train_dataset,
            )
    else:
        print(f"Experiment 3 not found")

    # 5. New: Experiment 4 visualizations (mirrors experiment 2)
    exp4_exists = False
    if seed_dir.exists():
        for sub in os.listdir(seed_dir):
            if sub.startswith('4-'):
                exp4_exists = True
                break

    if exp4_exists:
        print("Creating experiment 4 visualizations (aggregated across seeds)")

        # Gather eval datasets for exp4
        eval_datasets_exp4 = []
        exp4_name = None
        for exp in config.get('experiments', []):
            if exp.get('name', '').startswith('4-'):
                eval_datasets_exp4 = exp.get('evaluate_on', [])
                exp4_name = exp.get('name')
                break

        # Training dataset of exp4
        train_dataset_exp4 = None
        for exp in config.get('experiments', []):
            if exp.get('name') == exp4_name:
                train_dataset_exp4 = exp.get('train_on')
                break

        # Probe label mapping similar to exp2
        config_probes = []
        for arch in config.get('architectures', []):
            config_probes.append(arch.get('config_name', arch.get('name', '')))

        probe_labels_4 = {}
        for probe_name in config_probes:
            if 'sklearn_linear' in probe_name:
                agg_method = probe_name.split('_')[-1] if '_' in probe_name else 'mean'
                probe_labels_4[probe_name] = f'Linear ({agg_method})'
            elif 'act_sim' in probe_name:
                agg_method = probe_name.split('_')[-1] if '_' in probe_name else 'mean'
                probe_labels_4[probe_name] = f'Activation Similarity ({agg_method})'
            elif 'attention' in probe_name:
                probe_labels_4[probe_name] = 'Attention'
            elif 'sae' in probe_name:
                probe_labels_4[probe_name] = f'SAE ({probe_name})'
            else:
                probe_labels_4[probe_name] = probe_name

        # Evaluate for each dataset (ID + OOD) and both metrics
        for eval_dataset in eval_datasets_exp4:
            if train_dataset_exp4 is not None and eval_dataset == train_dataset_exp4:
                continue
            is_ood = (train_dataset_exp4 is not None and eval_dataset != train_dataset_exp4)
            title = "OOD" if is_ood else "Varying number of positive training examples\nwith 3000 negative examples"

            # Best defaults across types
            best_defaults_4 = get_best_default_probes_by_type(
                results_dir,
                args.seeds,
                exp_prefix='4-',
                
                eval_dataset=eval_dataset,
            )
            best_list_4 = list(best_defaults_4.values()) if best_defaults_4 else []
            for metric in ['auc', 'recall_at_fpr']:
                save_path = viz_root / f'experiment_4_best_defaults_comparison_{metric}_{eval_dataset}.png'
                if (not save_path.exists()) or args.force:
                    if best_list_4:
                        plot_experiment_unified(
                            results_dir,
                            best_list_4,
                            save_path=save_path,
                            metric=metric,
                            fpr_target=0.01,
                            
                            seeds=args.seeds,
                            plot_title=title,
                            eval_dataset=eval_dataset,
                            exp_prefix='4-',
                            probe_labels=probe_labels_4,
                            x_label=config.get('plot_labels', {}).get('x_label', None),
                            y_label=config.get('plot_labels', {}).get('y_label', None),
                        )
                else:
                    print(f"Skipping {save_path} (already exists, use --force to overwrite)")

        # Group comparisons for exp4 (ID only)
        patterns = default_probe_patterns()
        groups = {
            'sae': patterns['sae'],
            'lin_attn': patterns['linear'] + patterns['attention'],
            'act_sim': patterns['act_sim'],
        }
        for group_name, pats in groups.items():
            for metric in ['auc', 'recall_at_fpr']:
                save_path = viz_root / f'experiment_4_group_{group_name}_{metric}.png'
                if (not save_path.exists()) or args.force:
                    plot_probe_group_comparison(
                        results_dir,
                        pats,
                        save_path=save_path,
                        metric=metric,
                        fpr_target=0.01,
                        
                        seeds=args.seeds,
                        plot_title=None,
                        eval_dataset=train_dataset_exp4,
                        exp_prefix='4-',
                        require_default=True,
                        x_label=config.get('plot_labels', {}).get('x_label', None),
                        y_label=config.get('plot_labels', {}).get('y_label', None),
                    )
                else:
                    print(f"Skipping {save_path} (already exists, use --force to overwrite)")
    else:
        print("Experiment 4 not found")

    # 6. Group comparisons for experiment 2 (train set)
    if exp2_exists:
        patterns = default_probe_patterns()
        groups = {
            'sae': patterns['sae'],
            'lin_attn': patterns['linear'] + patterns['attention'],
            'act_sim': patterns['act_sim'],
        }
        for group_name, pats in groups.items():
            for metric in ['auc', 'recall_at_fpr']:
                save_path = viz_root / f'experiment_2_group_{group_name}_{metric}.png'
                if (not save_path.exists()) or args.force:
                    plot_probe_group_comparison(
                        results_dir,
                        pats,
                        save_path=save_path,
                        metric=metric,
                        fpr_target=0.01,
                        
                        seeds=args.seeds,
                        plot_title=None,
                        eval_dataset=train_dataset,
                        exp_prefix='2-',
                        require_default=True,
                        x_label=config.get('plot_labels', {}).get('x_label', None),
                        y_label=config.get('plot_labels', {}).get('y_label', None),
                    )
                else:
                    print(f"Skipping {save_path} (already exists, use --force to overwrite)")

    # 7. Scaling-law plots across Qwen runs for a representative default probe (linear_last)
    qwen_roots = []
    qwen_labels = []
    qwen_dirs = [
        ('spam_qwen_0.6B', '0.6B'),
        ('spam_qwen_1.7B', '1.7B'),
        ('spam_qwen_4B', '4B'),
        ('spam_qwen_8B', '8B'),
        ('spam_qwen_14B', '14B'),
        ('spam_qwen_32b', '32B'),
    ]
    for folder, label in qwen_dirs:
        root = Path('results') / folder
        if root.exists():
            qwen_roots.append(root)
            qwen_labels.append(label)
    if qwen_roots:
        for metric in ['auc', 'recall_at_fpr']:
            save_path = viz_root / f'scaling_linear_last_{metric}.png'
            if (not save_path.exists()) or args.force:
                plot_scaling_law_across_runs(
                    qwen_roots,
                    qwen_labels,
                    probe_pattern='linear_last',
                    save_path=save_path,
                    metric=metric,
                    fpr_target=0.01,
                    
                    seeds=args.seeds,
                    exp_prefix='2-',
                    x_label=config.get('plot_labels', {}).get('x_label', None),
                    y_label=config.get('plot_labels', {}).get('y_label', None),
                )
            else:
                print(f"Skipping {save_path} (already exists, use --force to overwrite)")

    # 8. Analysis CSVs for publication
    try:
        analysis.write_best_default_probe_tables(
            results_dir,
            viz_root,
            seeds=args.seeds,
            exp_prefixes=['2-', '4-'],
            filtered=args.filtered,
        )
        analysis.write_boost_and_best_tables(
            results_dir,
            viz_root,
            seeds=args.seeds,
            filtered=args.filtered,
            fpr_target=0.01,
        )
    except Exception as e:
        print(f"Analysis failed with error: {e}")


main()
