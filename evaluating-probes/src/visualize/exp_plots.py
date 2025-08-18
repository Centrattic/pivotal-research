from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import re

from .viz_core import (
    get_scores_and_labels,
    find_experiment_folders,
    extract_eval_and_train_from_filename,
    default_probe_patterns,
    collect_eval_result_files_for_pattern,
    auc as auc_metric,
    recall_at_fpr,
    draw_iqr_line,
    autoset_ylim_from_bands,
    find_inner_results_folder,
    collect_result_files_for_pattern,
    parse_llm_upsampling_from_filename,
)


def get_best_default_probes_by_type(
    base_results_dir: Path,
    seeds: List[str],
    exp_prefix: str,
    eval_dataset: Optional[str] = None,
) -> Dict[str, str]:
    patterns_per_type = default_probe_patterns()
    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        return {}
    exp_dir = experiment_dirs[0]

    best: Dict[str, str] = {}
    for probe_type, patterns in patterns_per_type.items():
        med_by_pattern: Dict[str, float] = {}
        for pattern in patterns:
            files = collect_eval_result_files_for_pattern(
                base_results_dir,
                seeds,
                exp_dir.name,
                evaluation_dirs=None,
                pattern=pattern,
                eval_dataset=eval_dataset,
                require_default=True,
            )
            aucs: List[float] = []
            for f in files:
                try:
                    scores, labels = get_scores_and_labels(f)
                    aucs.append(auc_metric(labels, scores))
                except Exception:
                    continue
            if aucs:
                med_by_pattern[pattern] = float(np.median(aucs))
        if med_by_pattern:
            best_pattern = max(med_by_pattern.items(), key=lambda kv: kv[1])[0]
            best[probe_type] = best_pattern
    return best


def plot_experiment_unified(
    base_results_dir: Path,
    probe_names: List[str],
    save_path=None,
    metric='auc',
    fpr_target: float = 0.01,
    seeds: List[str] = None,
    plot_title: str = None,
    eval_dataset: str = None,
    exp_prefix: str = '2-',
    probe_labels: Dict[str, str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    if seeds is None:
        seeds = ['42']
    if probe_labels is None:
        probe_labels = {p: p for p in probe_names}

    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        print(f"No experiment folders starting with {exp_prefix} found for seed {seeds[0]}")
        return
    exp_dir = experiment_dirs[0]

    plt.figure(figsize=(6, 4))
    colors = [f"C{i}" for i in range(len(probe_names))]
    is_ood = False
    global_lower_bounds: List[float] = []

    for probe_idx, probe_name in enumerate(probe_names):
        all_data: Dict[int, List[float]] = {}
        for seed in seeds:
            seed_dir = base_results_dir / f"seed_{seed}" / exp_dir.name
            if not seed_dir.exists():
                continue
            if eval_dataset:
                result_files = (
                    list((seed_dir / 'test_eval').glob(f"eval_on_{eval_dataset}__*{probe_name}*_results.json")) +
                    list((seed_dir / 'gen_eval').glob(f"eval_on_{eval_dataset}__*{probe_name}*_results.json"))
                )
            else:
                result_files = (
                    list((seed_dir / 'test_eval').glob(f"*{probe_name}*_results.json")) +
                    list((seed_dir / 'gen_eval').glob(f"*{probe_name}*_results.json"))
                )
            for file in result_files:
                file = str(file)
                m = re.search(r'class1_(\d+)', file)
                if not m:
                    try:
                        _, lbls = get_scores_and_labels(file)
                        n_pos = int(np.sum(np.array(lbls) == 1))
                    except Exception:
                        continue
                else:
                    n_pos = int(m.group(1))
                try:
                    scores, labels = get_scores_and_labels(file)
                    if metric == 'auc':
                        value = auc_metric(labels, scores)
                    else:
                        value = recall_at_fpr(labels, scores, fpr_target)
                    all_data.setdefault(n_pos, []).append(value)
                    if eval_dataset is not None:
                        eval_ds, train_ds = extract_eval_and_train_from_filename(file)
                        if eval_ds and train_ds and eval_ds != train_ds:
                            is_ood = True
                except Exception:
                    continue

        x_values = sorted(all_data.keys())
        if not x_values:
            continue
        medians, q25, q75 = [], [], []
        for x in x_values:
            vals = all_data[x]
            medians.append(np.median(vals))
            if len(vals) > 1:
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
            else:
                q25.append(vals[0])
                q75.append(vals[0])

        label = probe_labels.get(probe_name, probe_name)
        color = colors[probe_idx]
        ax = plt.gca()
        draw_iqr_line(ax, x_values, medians, q25, q75, label, color)
        if q25:
            global_lower_bounds.append(min(q25))

    if plot_title:
        plt.title(plot_title, fontsize=14)
    else:
        plt.title("Varying number of positive training examples\nwith 3000 negative examples", fontsize=14)

    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    if x_label is None:
        x_label = ("Number of positive examples in the train set" if not is_ood else "Number of positive training samples")
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_probe_group_comparison(
    base_results_dir: Path,
    probe_patterns: List[str],
    save_path=None,
    metric: str = 'auc',
    fpr_target: float = 0.01,
    seeds: List[str] = None,
    plot_title: str = None,
    eval_dataset: str = None,
    exp_prefix: str = '2-',
    probe_labels: Dict[str, str] = None,
    require_default: bool = True,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    import re
    if seeds is None:
        seeds = ['42']
    if probe_labels is None:
        probe_labels = {p: p for p in probe_patterns}

    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        print(f"No experiment folders starting with {exp_prefix} found for seed {seeds[0]}")
        return
    exp_dir = experiment_dirs[0]

    plt.figure(figsize=(6, 4))
    colors = [f"C{i}" for i in range(len(probe_patterns))]
    global_lower_bounds: List[float] = []

    for idx, pattern in enumerate(probe_patterns):
        files = collect_eval_result_files_for_pattern(
            base_results_dir,
            seeds,
            exp_dir.name,
            evaluation_dirs=None,
            pattern=pattern,
            eval_dataset=eval_dataset,
            require_default=require_default,
        )
        class1_to_values: Dict[int, List[float]] = {}
        for f in files:
            m = re.search(r'class1_(\d+)', f)
            if not m:
                try:
                    _, labels = get_scores_and_labels(f)
                    n_pos = int(np.sum(np.array(labels) == 1))
                except Exception:
                    continue
            else:
                n_pos = int(m.group(1))
            try:
                scores, labels = get_scores_and_labels(f)
                if metric == 'auc':
                    val = auc_metric(labels, scores)
                else:
                    val = recall_at_fpr(labels, scores, fpr_target)
                class1_to_values.setdefault(n_pos, []).append(val)
            except Exception:
                continue

        x_vals = sorted(k for k in class1_to_values.keys() if k is not None)
        if not x_vals:
            continue
        medians, q25, q75 = [], [], []
        for x in x_vals:
            vals = class1_to_values[x]
            medians.append(np.median(vals))
            if len(vals) > 1:
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
            else:
                q25.append(vals[0])
                q75.append(vals[0])

        color = colors[idx]
        label = probe_labels.get(pattern, pattern)
        draw_iqr_line(plt.gca(), x_vals, medians, q25, q75, label, color)
        if q25:
            global_lower_bounds.append(min(q25))

    if plot_title:
        plt.title(plot_title, fontsize=14)
    else:
        plt.title("Varying number of positive training examples\nwith 3000 negative examples", fontsize=14)
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_scaling_law_across_runs(
    run_roots: List[Path],
    run_labels: List[str],
    probe_pattern: str,
    save_path=None,
    metric: str = 'auc',
    fpr_target: float = 0.01,
    seeds: List[str] = None,
    exp_prefix: str = '2-',
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    import re
    if seeds is None:
        seeds = ['42']
    plt.figure(figsize=(6, 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(run_roots)))
    global_lower_bounds: List[float] = []

    for idx, (root, label) in enumerate(zip(run_roots, run_labels)):
        exp_dirs = find_experiment_folders(root, seeds[0], exp_prefix)
        if not exp_dirs:
            continue
        exp_dir = exp_dirs[0]
        files = collect_eval_result_files_for_pattern(
            root, seeds, exp_dir.name, evaluation_dirs=None, pattern=probe_pattern, eval_dataset=None, require_default=True
        )
        class1_to_values: Dict[int, List[float]] = {}
        for f in files:
            m = re.search(r'class1_(\d+)', f)
            if not m:
                continue
            n_pos = int(m.group(1))
            try:
                scores, labels = get_scores_and_labels(f)
                if metric == 'auc':
                    val = auc_metric(labels, scores)
                else:
                    val = recall_at_fpr(labels, scores, fpr_target)
                class1_to_values.setdefault(n_pos, []).append(val)
            except Exception:
                continue

        x_vals = sorted(class1_to_values.keys())
        if not x_vals:
            continue
        medians, q25, q75 = [], [], []
        for x in x_vals:
            vals = class1_to_values[x]
            medians.append(np.median(vals))
            if len(vals) > 1:
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
            else:
                q25.append(vals[0])
                q75.append(vals[0])
        color = colors[idx]
        draw_iqr_line(plt.gca(), x_vals, medians, q25, q75, label, color)
        if q25:
            global_lower_bounds.append(min(q25))

    plt.title("Scaling law across model sizes", fontsize=14)
    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_scaling_law_all_probes_aggregated(
    run_roots: List[Path],
    run_labels: List[str],
    seeds: List[str],
    exp_prefix: str,
    aggregated_out_dir: Path,
    fpr_target: float = 0.01,
):
    """
    Generate scaling law plots for ALL default probe patterns across runs and
    save ONLY to the aggregated visualizations directory.
    Produces both AUC and Recall@FPR plots for each probe pattern.
    """
    import os
    os.makedirs(aggregated_out_dir, exist_ok=True)

    patterns = []
    for plist in default_probe_patterns().values():
        patterns.extend(plist)
    # Deduplicate while preserving order
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]

    for pattern in patterns:
        # AUC plot
        auc_path = aggregated_out_dir / f"scaling_{pattern}_auc.png"
        plot_scaling_law_across_runs(
            run_roots=run_roots,
            run_labels=run_labels,
            probe_pattern=pattern,
            save_path=str(auc_path),
            metric='auc',
            fpr_target=fpr_target,
            seeds=seeds,
            exp_prefix=exp_prefix,
        )
        # Recall@FPR plot
        rec_path = aggregated_out_dir / f"scaling_{pattern}_recall_at_{int(fpr_target*10000)}ppm_fpr.png"
        plot_scaling_law_across_runs(
            run_roots=run_roots,
            run_labels=run_labels,
            probe_pattern=pattern,
            save_path=str(rec_path),
            metric='recall',
            fpr_target=fpr_target,
            seeds=seeds,
            exp_prefix=exp_prefix,
        )



def plot_llm_upsampling_per_probe(
    base_results_dir: Path,
    probe_pattern: str,
    save_path=None,
    metric: str = 'auc',
    fpr_target: float = 0.01,
    seeds: List[str] = None,
    eval_dataset: str = None,
    exp_prefix: str = '3-',
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    """Plot performance vs. number of real positive samples for different LLM upsampling factors, per probe.

    One line per upsampling factor; shaded IQR across seeds. Defaults-only files are automatically chosen.
    """
    import re
    if seeds is None:
        seeds = ['42']

    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        print(f"No experiment folders starting with {exp_prefix} found for seed {seeds[0]}")
        return
    exp_dir = experiment_dirs[0]
    inner = find_inner_results_folder(exp_dir)
    if inner is None:
        print(f"No inner results folder found under {exp_dir}")
        return

    files = collect_result_files_for_pattern(
        base_results_dir,
        seeds,
        exp_dir.name,
        inner.name,
        probe_pattern,
        eval_dataset=eval_dataset,
        require_default=True,
    )
    if not files:
        print(f"No files for probe pattern {probe_pattern} in {exp_dir}")
        return

    # Group values by upsampling factor, then by number of real positives
    from collections import defaultdict
    values_by_factor: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        n_real, factor = parse_llm_upsampling_from_filename(f)
        if n_real is None:
            # try to fall back to class1 capture
            m = re.search(r'class1_(\d+)', f)
            if not m:
                continue
            n_real = int(m.group(1))
            factor = 1
        try:
            scores, labels = get_scores_and_labels(f)
            if metric == 'auc':
                val = auc_metric(labels, scores)
            else:
                val = recall_at_fpr(labels, scores, fpr_target)
            values_by_factor[factor][n_real].append(val)
        except Exception:
            continue

    if not values_by_factor:
        return

    plt.figure(figsize=(6, 4))
    # Sort factors with 1 first, then ascending
    sorted_factors = sorted(values_by_factor.keys())
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, max(2, len(sorted_factors))))
    global_lower_bounds: List[float] = []

    for idx, factor in enumerate(sorted_factors):
        inner_map = values_by_factor[factor]
        x_vals = sorted(inner_map.keys())
        if not x_vals:
            continue
        medians, q25, q75 = [], [], []
        for x in x_vals:
            vals = inner_map[x]
            medians.append(np.median(vals))
            if len(vals) > 1:
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
            else:
                q25.append(vals[0])
                q75.append(vals[0])
        label = (f"no upsampling" if factor == 1 else f"upsample x{factor}")
        draw_iqr_line(plt.gca(), x_vals, medians, q25, q75, label, colors[idx])
        if q25:
            global_lower_bounds.append(min(q25))

    title = f"LLM upsampling per probe: {probe_pattern}"
    plt.title(title, fontsize=14)
    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
