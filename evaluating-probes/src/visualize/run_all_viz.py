import argparse
import yaml
import os
from pathlib import Path
import glob
import re
from src.visualize.utils_viz import (
    plot_experiment_unified,
    plot_probe_group_comparison,
    get_best_default_probes_by_type,
    default_probe_patterns,
    plot_scaling_law_across_runs,
    plot_scaling_law_all_probes_aggregated,
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
    # Aggregated cross-run visualizations folder
    agg_viz_root = Path('results') / '_aggregated' / 'visualizations'
    agg_viz_root.mkdir(
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

    # 1. Identify experiment folders under the first seed
    exp_folders = {k: None for k in ['2', '3', '4']}
    first_seed = args.seeds[0]
    seed_dir = results_dir / f"seed_{first_seed}"
    if seed_dir.exists():
        for sub in os.listdir(seed_dir):
            for k in exp_folders:
                if sub.startswith(f'{k}-'):
                    exp_folders[k] = sub
                    break
    exp_folders = {k: v for k, v in exp_folders.items() if v is not None}

    # 2. (removed) legacy CSV histograms and multi-folder plots for simplicity

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

    # 3. (optional) Experiment 3 per-probe plots are omitted in the simplified pipeline
    if not exp3_exists:
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

    # 6. Scaling-law plots across Qwen runs
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
        # Per-probe scaling law for a representative probe (linear_last)
        for metric in ['auc', 'recall_at_fpr']:
            agg_save_path = agg_viz_root / f'scaling_linear_last_{metric}.png'
            if (not agg_save_path.exists()) or args.force:
                plot_scaling_law_across_runs(
                    qwen_roots,
                    qwen_labels,
                    probe_pattern='linear_last',
                    save_path=agg_save_path,
                    metric=metric,
                    fpr_target=0.01,
                    seeds=args.seeds,
                    exp_prefix='2-',
                    x_label=config.get('plot_labels', {}).get('x_label', None),
                    y_label=config.get('plot_labels', {}).get('y_label', None),
                )
        # Aggregated scaling laws for ALL default probes, saved only to aggregated folder
        plot_scaling_law_all_probes_aggregated(
            qwen_roots,
            qwen_labels,
            args.seeds,
            exp_prefix='2-',
            aggregated_out_dir=agg_viz_root,
            fpr_target=0.01,
        )

    # 8. Analysis CSVs for publication
    try:
        analysis.write_best_default_probe_tables(
            results_dir,
            viz_root,
            seeds=args.seeds,
            exp_prefixes=['2-', '4-'],
        )
        analysis.write_boost_and_best_tables(
            results_dir,
            viz_root,
            seeds=args.seeds,
            fpr_target=0.01,
        )
    except Exception as e:
        print(f"Analysis failed with error: {e}")


main()
