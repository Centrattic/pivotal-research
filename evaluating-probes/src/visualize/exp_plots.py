from pathlib import Path
from typing import Dict, List, Optional

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
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
    paper_label_for_probe,
)


def _load_cache(cache_dir: Path) -> Dict:
    try:
        path = cache_dir / 'metrics_cache.json'
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


def _save_cache(cache_dir: Path, data: Dict) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / 'metrics_cache.json'
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

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
    verbose: bool = False,
):
    if seeds is None:
        seeds = ['42']
    if probe_labels is None:
        probe_labels = {p: p for p in probe_names}

    # Cache: use same folder as save_path if provided
    cache_dir: Optional[Path] = None
    if save_path is not None:
        try:
            cache_dir = Path(save_path).parent
        except Exception:
            cache_dir = None
    cache_key = None
    if cache_dir is not None:
        cache = _load_cache(cache_dir)
        cache_key = (
            f"unified|exp:{exp_prefix}|metric:{metric}|eval:{eval_dataset or 'ALL'}|"
            f"seeds:{','.join(seeds)}|probes:{','.join(sorted(probe_names))}"
        )
        cached = cache.get(cache_key)
        if cached:
            if verbose:
                print(f"[cache] hit: {cache_key}", flush=True)
            # Render from cache
            plt.figure(figsize=(6, 4))
            colors = [f"C{i}" for i in range(len(probe_names))]
            global_lower_bounds: List[float] = []
            for idx, probe_name in enumerate(probe_names):
                series = cached.get(probe_name)
                if not series:
                    continue
                x_values = series.get('n', [])
                med = series.get('med', [])
                q25 = series.get('q25', [])
                q75 = series.get('q75', [])
                if not x_values:
                    continue
                label = probe_labels.get(probe_name, paper_label_for_probe(probe_name))
                draw_iqr_line(plt.gca(), x_values, med, q25, q75, label, colors[idx])
                if q25:
                    global_lower_bounds.append(min(q25))
            if plot_title:
                plt.title(plot_title, fontsize=14)
            else:
                plt.title("Varying number of positive training examples\nwith 3000 negative examples", fontsize=14)
            if y_label is None:
                y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
            if x_label is None:
                x_label = ("Number of positive examples in the train set")
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
            return

    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        print(f"No experiment folders starting with {exp_prefix} found for seed {seeds[0]}")
        return
    exp_dir = experiment_dirs[0]
    if verbose:
        print(f"[unified] Using experiment folder: {exp_dir}", flush=True)

    plt.figure(figsize=(6, 4))
    colors = [f"C{i}" for i in range(len(probe_names))]
    is_ood = False
    global_lower_bounds: List[float] = []

    # Prepare container for caching
    to_cache: Dict[str, Dict[str, List[float]]] = {}

    for probe_idx, probe_name in enumerate(probe_names):
        all_data: Dict[int, List[float]] = {}
        if verbose:
            print(f"[unified] Probe: {probe_name}", flush=True)
        # Use folder-agnostic collector with default-only filtering
        result_files = collect_eval_result_files_for_pattern(
            base_results_dir,
            seeds,
            exp_dir.name,
            evaluation_dirs=None,
            pattern=probe_name,
            eval_dataset=eval_dataset,
            require_default=True,
        )
        if verbose:
            print(f"[unified]  collected files={len(result_files)}", flush=True)
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
            if verbose:
                print(f"[unified]  No x-values for probe {probe_name}", flush=True)
            continue
        medians, q25, q75 = [], [], []
        for x in x_values:
            vals = all_data[x]
            medians.append(np.median(vals))
            if len(vals) > 1:
                q25.append(np.percentile(vals, 5))
                q75.append(np.percentile(vals, 95))
            else:
                q25.append(vals[0])
                q75.append(vals[0])
        if verbose:
            print(f"[unified]  points={len(x_values)}", flush=True)

        label = probe_labels.get(probe_name, paper_label_for_probe(probe_name))
        color = colors[probe_idx]
        ax = plt.gca()
        draw_iqr_line(ax, x_values, medians, q25, q75, label, color)
        if q25:
            global_lower_bounds.append(min(q25))
        # add to cache series
        to_cache[probe_name] = {
            'n': x_values,
            'med': medians,
            'q25': q25,
            'q75': q75,
        }

    if plot_title:
        from .viz_core import wrap_text_for_plot
        plt.title(wrap_text_for_plot(plot_title), fontsize=14)
    else:
        plt.title("Varying number of positive training examples\nwith 3000 negative examples", fontsize=14)

    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    if x_label is None:
        x_label = ("Number of positive examples in the train set" if not is_ood else "Number of positive training samples")
    from .viz_core import wrap_text_for_plot
    plt.ylabel(wrap_text_for_plot(y_label), fontsize=12)
    plt.xlabel(wrap_text_for_plot(x_label), fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        if verbose:
            print(f"[unified] Saving figure to {save_path}", flush=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        if cache_dir is not None and cache_key is not None:
            cache = _load_cache(cache_dir)
            cache[cache_key] = to_cache
            _save_cache(cache_dir, cache)
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
    verbose: bool = False,
):
    import re
    if seeds is None:
        seeds = ['42']
    if probe_labels is None:
        probe_labels = {p: p for p in probe_patterns}

    # Cache
    cache_dir: Optional[Path] = None
    if save_path is not None:
        try:
            cache_dir = Path(save_path).parent
        except Exception:
            cache_dir = None
    cache_key = None
    if cache_dir is not None:
        cache = _load_cache(cache_dir)
        cache_key = (
            f"group|exp:{exp_prefix}|metric:{metric}|eval:{eval_dataset or 'ALL'}|"
            f"seeds:{','.join(seeds)}|patterns:{','.join(sorted(probe_patterns))}"
        )
        cached = cache.get(cache_key)
        if cached:
            if verbose:
                print(f"[cache] hit: {cache_key}", flush=True)
            plt.figure(figsize=(6, 4))
            colors = [f"C{i}" for i in range(len(probe_patterns))]
            global_lower_bounds: List[float] = []
            for idx, pattern in enumerate(probe_patterns):
                series = cached.get(pattern)
                if not series:
                    continue
                x_vals = series.get('n', [])
                med = series.get('med', [])
                q25 = series.get('q25', [])
                q75 = series.get('q75', [])
                draw_iqr_line(plt.gca(), x_vals, med, q25, q75, probe_labels.get(pattern, paper_label_for_probe(pattern)), colors[idx])
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
            return
    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        print(f"No experiment folders starting with {exp_prefix} found for seed {seeds[0]}")
        return
    exp_dir = experiment_dirs[0]
    if verbose:
        print(f"[group] Using experiment folder: {exp_dir}", flush=True)

    plt.figure(figsize=(6, 4))
    colors = [f"C{i}" for i in range(len(probe_patterns))]
    global_lower_bounds: List[float] = []

    to_cache: Dict[str, Dict[str, List[float]]] = {}
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
        if verbose:
            print(f"[group] Pattern={pattern} files={len(files)}", flush=True)
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
            if verbose:
                print(f"[group] Pattern={pattern} no x-values", flush=True)
            continue
        medians, q25, q75 = [], [], []
        for x in x_vals:
            vals = class1_to_values[x]
            medians.append(np.median(vals))
            if len(vals) > 1:
                q25.append(np.percentile(vals, 5))
                q75.append(np.percentile(vals, 95))
            else:
                q25.append(vals[0])
                q75.append(vals[0])
        if verbose:
            print(f"[group] Pattern={pattern} points={len(x_vals)}", flush=True)

        color = colors[idx]
        label = probe_labels.get(pattern, paper_label_for_probe(pattern))
        draw_iqr_line(plt.gca(), x_vals, medians, q25, q75, label, color)
        if q25:
            global_lower_bounds.append(min(q25))
        to_cache[pattern] = {
            'n': x_vals,
            'med': medians,
            'q25': q25,
            'q75': q75,
        }

    if plot_title:
        from .viz_core import wrap_text_for_plot
        plt.title(wrap_text_for_plot(plot_title), fontsize=14)
    else:
        plt.title("Varying number of positive training examples\nwith 3000 negative examples", fontsize=14)
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    from .viz_core import wrap_text_for_plot
    plt.xlabel(wrap_text_for_plot(x_label), fontsize=12)
    plt.ylabel(wrap_text_for_plot(y_label), fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        if verbose:
            print(f"[group] Saving figure to {save_path}", flush=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        if cache_dir is not None and cache_key is not None:
            cache = _load_cache(cache_dir)
            cache[cache_key] = to_cache
            _save_cache(cache_dir, cache)
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
    verbose: bool = False,
    require_in_distribution: bool = False,
    cache_dir: Optional[Path] = None,
):
    import re
    if seeds is None:
        seeds = ['42']
    plt.figure(figsize=(6, 4))
    # Stronger blue gradient for scaling
    colors = plt.cm.Blues(np.linspace(0.2, 0.98, len(run_roots)))
    global_lower_bounds: List[float] = []

    if verbose:
        print(f"[scaling] Runs={len(run_roots)} labels={run_labels}", flush=True)

    # Try cache
    if cache_dir is None and save_path is not None:
        cache_dir = Path(save_path).parent
    cache_key = None
    cache = {}
    if cache_dir is not None:
        cache = _load_cache(cache_dir)
        cache_key = (
            f"scaling|probe:{probe_pattern}|metric:{metric}|fpr:{fpr_target}|exp:{exp_prefix}|"
            f"seeds:{','.join(seeds or ['42'])}|runs:{','.join(run_labels)}|id:{int(require_in_distribution)}"
        )
        if cache_key in cache:
            data = cache.get(cache_key, {})
            # Validate cached shape before using
            if isinstance(data, dict) and any(k in data for k in run_labels):
                if verbose:
                    print("[scaling] Using cached medians/IQRs", flush=True)
                for idx, label in enumerate(run_labels):
                    series = data.get(label)
                    if not series:
                        continue
                    x_vals = series.get('n', [])
                    med = series.get('med', [])
                    q25 = series.get('q25', [])
                    q75 = series.get('q75', [])
                    if not x_vals:
                        continue
                    color = plt.cm.Blues(np.linspace(0.2, 0.98, len(run_roots)))[idx]
                    draw_iqr_line(plt.gca(), x_vals, med, q25, q75, label, color)
                plt.title(f"Scaling across positive samples — {paper_label_for_probe(probe_pattern)}", fontsize=14)
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
                if save_path:
                    plt.savefig(save_path, dpi=150)
                    plt.close()
                else:
                    plt.show()
                return
    per_label_series: Dict[str, Dict[str, List[float]]] = {}
    for idx, (root, label) in enumerate(zip(run_roots, run_labels)):
        exp_dirs = find_experiment_folders(root, seeds[0], exp_prefix)
        if not exp_dirs:
            continue
        exp_dir = exp_dirs[0]
        files = collect_eval_result_files_for_pattern(
            root, seeds, exp_dir.name, evaluation_dirs=None, pattern=probe_pattern, eval_dataset=None, require_default=True
        )
        if require_in_distribution:
            # Keep only files where eval_on == train_on (ID results)
            filtered = []
            for f in files:
                ev, tr = extract_eval_and_train_from_filename(f)
                if ev is not None and tr is not None and ev == tr:
                    filtered.append(f)
            if verbose:
                print(f"[scaling] label={label} filtered to ID files={len(filtered)}", flush=True)
            files = filtered
            if not files:
                if verbose:
                    print(f"[scaling] No ID files found for {label}; falling back to all eval files", flush=True)
                # Recollect unfiltered files (already in 'files' before), so do nothing as we can re-query quickly
                files = collect_eval_result_files_for_pattern(
                    root, seeds, exp_dir.name, evaluation_dirs=None, pattern=probe_pattern, eval_dataset=None, require_default=True
                )
        if verbose:
            print(f"[scaling] label={label} files={len(files)}", flush=True)
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
                q25.append(np.percentile(vals, 5))
                q75.append(np.percentile(vals, 95))
            else:
                q25.append(vals[0])
                q75.append(vals[0])
        color = colors[idx]
        draw_iqr_line(plt.gca(), x_vals, medians, q25, q75, label, color)
        if q25:
            global_lower_bounds.append(min(q25))
        # Collect for cache
        per_label_series[label] = {
            'n': x_vals,
            'med': medians,
            'q25': q25,
            'q75': q75,
        }

    probe_readable = paper_label_for_probe(probe_pattern)
    from .viz_core import wrap_text_for_plot
    plt.title(wrap_text_for_plot(f"Scaling across positive samples — {probe_readable}"), fontsize=14)
    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    from .viz_core import wrap_text_for_plot
    plt.ylabel(wrap_text_for_plot(y_label), fontsize=12)
    plt.xlabel(wrap_text_for_plot(x_label), fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        if verbose:
            print(f"[scaling] Saving figure to {save_path}", flush=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        # Save cache (only if non-empty and valid)
        if cache_dir is not None and cache_key is not None and per_label_series:
            cache = _load_cache(cache_dir)
            cache[cache_key] = per_label_series
            _save_cache(cache_dir, cache)
    else:
        plt.show()


def plot_scaling_law_all_probes_aggregated(
    run_roots: List[Path],
    run_labels: List[str],
    seeds: List[str],
    exp_prefix: str,
    aggregated_out_dir: Path,
    fpr_target: float = 0.01,
    verbose: bool = False,
    probe_patterns: Optional[Dict[str, List[str]]] = None,
    generate_per_probe_plots: bool = False,
):
    """
    Generate scaling law plots for ALL default probe patterns across runs and
    save ONLY to the aggregated visualizations directory.
    Produces both AUC and Recall@FPR plots for each probe pattern.
    """
    import os
    os.makedirs(aggregated_out_dir, exist_ok=True)

    patterns = []
    if probe_patterns is None:
        for plist in default_probe_patterns().values():
            patterns.extend(plist)
    else:
        for plist in probe_patterns.values():
            patterns.extend(plist)
    # Deduplicate while preserving order
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]

    if generate_per_probe_plots:
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
                verbose=verbose,
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
                verbose=verbose,
            )

    # After per-probe scaling plots, add n=1 heatmaps across probes and sizes
    try:
        plot_n1_heatmaps_across_runs(
            run_roots=run_roots,
            run_labels=run_labels,
            seeds=seeds,
            exp_prefix=exp_prefix,
            aggregated_out_dir=aggregated_out_dir,
            fpr_target=fpr_target,
            verbose=verbose,
            probe_patterns=probe_patterns,
        )
        # OOD heatmaps for 87_is_spam
        plot_n1_heatmaps_across_runs(
            run_roots=run_roots,
            run_labels=run_labels,
            seeds=seeds,
            exp_prefix=exp_prefix,
            aggregated_out_dir=aggregated_out_dir,
            fpr_target=fpr_target,
            verbose=verbose,
            probe_patterns=probe_patterns,
            eval_dataset='87_is_spam',
        )
    except Exception as e:
        if verbose:
            print(f"[heatmap] Failed to render n=1 heatmaps: {e}", flush=True)


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
    verbose: bool = False,
    plot_title: Optional[str] = None,
):
    """Plot performance vs. number of real positive samples for different LLM upsampling factors, per probe.

    One line per upsampling factor; shaded IQR across seeds. Defaults-only files are automatically chosen.
    """
    import re
    if seeds is None:
        seeds = ['42']

    # Cache
    cache_dir: Optional[Path] = None
    if save_path is not None:
        try:
            cache_dir = Path(save_path).parent
        except Exception:
            cache_dir = None
    cache_key = None
    if cache_dir is not None:
        cache = _load_cache(cache_dir)
        cache_key = (
            f"llm|exp:{exp_prefix}|metric:{metric}|eval:{eval_dataset or 'ALL'}|"
            f"seeds:{','.join(seeds)}|probe:{probe_pattern}|fpr:{fpr_target}"
        )
        cached = cache.get(cache_key)
        if cached:
            if verbose:
                print(f"[cache] hit: {cache_key}", flush=True)
            plt.figure(figsize=(6, 4))
            colors = plt.cm.Reds(np.linspace(0.2, 0.98, max(2, len(cached))))
            global_lower_bounds: List[float] = []
            for idx, item in enumerate(sorted(cached.items(), key=lambda kv: int(kv[0]))):
                factor, series = item
                x_vals = series.get('n', [])
                med = series.get('med', [])
                q25 = series.get('q25', [])
                q75 = series.get('q75', [])
                draw_iqr_line(plt.gca(), x_vals, med, q25, q75, ("no upsampling" if int(factor) == 1 else f"upsample x{factor}"), colors[idx])
                if q25:
                    global_lower_bounds.append(min(q25))
            probe_readable = paper_label_for_probe(probe_pattern)
            title = f"LLM upsampling — {probe_readable}"
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
            return

    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        print(f"No experiment folders starting with {exp_prefix} found for seed {seeds[0]}")
        return
    exp_dir = experiment_dirs[0]
    inner = find_inner_results_folder(exp_dir)
    if inner is None:
        print(f"No inner results folder found under {exp_dir}")
        return

    # For upsampling, search in test/gen eval regardless of presence of 'trained' folder
    files = collect_eval_result_files_for_pattern(
        base_results_dir,
        seeds,
        exp_dir.name,
        evaluation_dirs=None,
        pattern=probe_pattern,
        eval_dataset=eval_dataset,
        require_default=True,
    )
    if not files:
        print(f"No files for probe pattern {probe_pattern} in {exp_dir}")
        return

    # Group values by upsampling factor, then by number of real positives
    from collections import defaultdict
    values_by_factor: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    files_by_factor: Dict[int, List[str]] = defaultdict(list)
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
            files_by_factor[factor].append(f)
        except Exception:
            continue

    if not values_by_factor:
        return

    if verbose:
        print(f"[llm] total_files={len(files)} for pattern={probe_pattern}", flush=True)
        print("[llm] listing files used:", flush=True)
        for f in sorted(files):
            print(f"   {f}", flush=True)
    plt.figure(figsize=(6, 4))
    # Sort factors with 1 first, then ascending
    sorted_factors = sorted(values_by_factor.keys())
    # Stronger red/orange gradient for upsampling
    colors = plt.cm.Reds(np.linspace(0.2, 0.98, max(2, len(sorted_factors))))
    global_lower_bounds: List[float] = []

    to_cache: Dict[str, Dict[str, List[float]]] = {}
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
                q25.append(np.percentile(vals, 5))
                q75.append(np.percentile(vals, 95))
            else:
                q25.append(vals[0])
                q75.append(vals[0])
        if verbose:
            print(f"[llm] factor={factor} points={len(x_vals)}", flush=True)
            ff = sorted(files_by_factor.get(factor, []))
            print(f"[llm]   files for factor {factor} (n={len(ff)}):", flush=True)
            for path in ff:
                print(f"      {path}", flush=True)
        # Do not use legend labels; color encodes factor
        draw_iqr_line(plt.gca(), x_vals, medians, q25, q75, None, colors[idx])
        if q25:
            global_lower_bounds.append(min(q25))
        to_cache[str(factor)] = {
            'n': x_vals,
            'med': medians,
            'q25': q25,
            'q75': q75,
        }

    probe_readable = paper_label_for_probe(probe_pattern)
    if plot_title is None:
        title = f"LLM upsampling — {probe_readable}"
    else:
        title = plot_title
    from .viz_core import wrap_text_for_plot
    plt.title(wrap_text_for_plot(title), fontsize=14)
    if y_label is None:
        y_label = ("AUC" if metric == 'auc' else f"Recall at {fpr_target*100}% FPR")
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    from .viz_core import wrap_text_for_plot
    plt.ylabel(wrap_text_for_plot(y_label), fontsize=12)
    plt.xlabel(wrap_text_for_plot(x_label), fontsize=12)
    plt.xscale('log')
    # Replace legend with a colorbar for upsampling factors
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.Reds, norm=mpl.colors.Normalize(vmin=min(sorted_factors), vmax=max(sorted_factors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('LLM upsampling factor', fontsize=10)
    cbar.set_ticks([1, 2, 3, 4, 5, 10, 20])
    cbar.set_ticklabels(['1', '2', '3', '4', '5', '10', '20'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    autoset_ylim_from_bands(plt.gca(), global_lower_bounds, pad=0.05)
    if save_path:
        if verbose:
            print(f"[llm] Saving figure to {save_path}", flush=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        if cache_dir is not None and cache_key is not None:
            cache = _load_cache(cache_dir)
            cache[cache_key] = to_cache
            _save_cache(cache_dir, cache)
    else:
        plt.show()


def plot_n1_heatmaps_across_runs(
    run_roots: List[Path],
    run_labels: List[str],
    seeds: List[str],
    exp_prefix: str,
    aggregated_out_dir: Path,
    fpr_target: float = 0.01,
    verbose: bool = False,
    probe_patterns: Optional[Dict[str, List[str]]] = None,
    eval_dataset: Optional[str] = None,
):
    """Two heatmaps (AUC and Recall@FPR) for n=1 across probe groups and model sizes.
    Rows are grouped (merged) by normalized paper labels, e.g.,
    SAE (mean/max/last/softmax) are lumped across model-specific SAE configs.
    Columns are model sizes (one per run in run_labels).
    """
    import os
    import re
    # Build flat list of patterns then group them by normalized label
    patterns: List[str] = []
    if probe_patterns is None:
        for plist in default_probe_patterns().values():
            patterns.extend(plist)
    else:
        for plist in probe_patterns.values():
            patterns.extend(plist)
    # Dedup
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]

    # Helper to normalize labels so SAE variants across models collapse
    def _normalize_label(label: str) -> str:
        if label.startswith('SAE ('):
            # Map any model/width-specific SAE label to pool-only label
            if 'mean' in label:
                return 'SAE (mean pool)'
            if 'max' in label:
                return 'SAE (max pool)'
            if 'last token' in label or 'last' in label:
                return 'SAE (last token)'
            if 'softmax' in label:
                return 'SAE (softmax)'
            return 'SAE'
        return label

    # Group patterns by normalized paper label
    from .viz_core import paper_label_for_probe
    grouped: Dict[str, List[str]] = {}
    for pat in patterns:
        lab = _normalize_label(paper_label_for_probe(pat))
        grouped.setdefault(lab, []).append(pat)

    group_labels = list(grouped.keys())
    auc_mat = np.full((len(group_labels), len(run_roots)), np.nan, dtype=float)
    rec_mat = np.full((len(group_labels), len(run_roots)), np.nan, dtype=float)

    for c_idx, (root, label) in enumerate(zip(run_roots, run_labels)):
        exp_dirs = find_experiment_folders(root, seeds[0], exp_prefix)
        if not exp_dirs:
            continue
        exp_dir = exp_dirs[0]
        for r_idx, (g_label, pat_list) in enumerate(grouped.items()):
            vals_auc: List[float] = []
            vals_rec: List[float] = []
            # Union of files across all patterns in the group
            for pattern in pat_list:
                files = collect_eval_result_files_for_pattern(
                    root, seeds, exp_dir.name, evaluation_dirs=None, pattern=pattern, eval_dataset=eval_dataset, require_default=True
                )
                files = [f for f in files if re.search(r'class1_1(?!\d)', f)]
                for f in files:
                    try:
                        scores, labels = get_scores_and_labels(f)
                        vals_auc.append(auc_metric(labels, scores))
                        vals_rec.append(recall_at_fpr(labels, scores, fpr_target))
                    except Exception:
                        continue
            if vals_auc:
                auc_mat[r_idx, c_idx] = float(np.median(vals_auc))
            if vals_rec:
                rec_mat[r_idx, c_idx] = float(np.median(vals_rec))

    os.makedirs(aggregated_out_dir, exist_ok=True)
    # Try to cache heatmap matrices
    cache = _load_cache(aggregated_out_dir)
    eval_suffix = f"|eval:{eval_dataset}" if eval_dataset else ""
    # Bump schema to invalidate old cached matrices that used un-grouped labels
    schema_suffix = "|schema:v2_grouped_rows"
    cache_key_auc = f"heatmap_n1_auc|exp:{exp_prefix}|seeds:{','.join(seeds)}|runs:{','.join(run_labels)}|fpr:{fpr_target}{eval_suffix}{schema_suffix}"
    cache_key_rec = f"heatmap_n1_rec|exp:{exp_prefix}|seeds:{','.join(seeds)}|runs:{','.join(run_labels)}|fpr:{fpr_target}{eval_suffix}{schema_suffix}"
    cached_auc = cache.get(cache_key_auc)
    cached_rec = cache.get(cache_key_rec)
    if cached_auc is not None and cached_rec is not None:
        auc_mat = np.array(cached_auc, dtype=float)
        rec_mat = np.array(cached_rec, dtype=float)
    # Optionally drop rows that are all-NaN to avoid empty stripes
    def _drop_all_nan_rows(mat: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
        keep_idx = [i for i in range(mat.shape[0]) if not np.all(np.isnan(mat[i, :]))]
        if not keep_idx:
            return mat, labels
        return mat[keep_idx, :], [labels[i] for i in keep_idx]

    def _heatmap(mat: np.ndarray, title: str, out_path: Path):
        plt.figure(figsize=(6, 4))
        data, labels_rows = _drop_all_nan_rows(mat, group_labels)
        im = plt.imshow(data, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0, cmap='viridis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.yticks(ticks=list(range(len(labels_rows))), labels=labels_rows, fontsize=8)
        plt.xticks(ticks=list(range(len(run_labels))), labels=run_labels)
        from .viz_core import wrap_text_for_plot
        plt.title(wrap_text_for_plot(title), fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    # Generate appropriate filenames based on eval_dataset
    suffix = f"_ood_{eval_dataset}" if eval_dataset else ""
    _heatmap(auc_mat, f'AUC (n=1){suffix}', aggregated_out_dir / f'heatmap_n1_auc{suffix}.png')
    _heatmap(rec_mat, f'Recall at {fpr_target*100}% FPR (n=1){suffix}', aggregated_out_dir / f'heatmap_n1_recall_at_{int(fpr_target*10000)}ppm_fpr{suffix}.png')
    # Save to cache for next time
    cache = _load_cache(aggregated_out_dir)
    cache[cache_key_auc] = auc_mat.tolist()
    cache[cache_key_rec] = rec_mat.tolist()
    _save_cache(aggregated_out_dir, cache)


def plot_scaling_law_aggregated_across_probes(
    run_roots: List[Path],
    run_labels: List[str],
    seeds: List[str],
    exp_prefix: str,
    save_path=None,
    metric: str = 'auc',
    fpr_target: float = 0.01,
    verbose: bool = False,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    probe_patterns: Optional[Dict[str, List[str]]] = None,
    eval_dataset: Optional[str] = None,
):
    """
    Generate scaling law plot showing median scores across ALL probe types.
    No error bars or shaded regions - just median lines.
    """
    if seeds is None:
        seeds = ['42']
    
    plt.figure(figsize=(6, 4))
    colors = plt.cm.Blues(np.linspace(0.2, 0.98, len(run_roots)))
    
    if verbose:
        print(f"[scaling_agg] Runs={len(run_roots)} labels={run_labels}", flush=True)
    
    # Get all probe patterns - use provided patterns or fall back to defaults
    if probe_patterns is None:
        patterns = []
        for plist in default_probe_patterns().values():
            patterns.extend(plist)
    else:
        patterns = []
        for plist in probe_patterns.values():
            patterns.extend(plist)
    # Deduplicate while preserving order
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]
    
    if verbose:
        print(f"[scaling_agg] Using {len(patterns)} probe patterns: {patterns}", flush=True)
    
    # Try to load cache
    cache_dir = None
    if save_path is not None:
        cache_dir = Path(save_path).parent
    cache = {}
    cache_key = None
    if cache_dir is not None:
        cache = _load_cache(cache_dir)
        eval_suffix = f"|eval:{eval_dataset}" if eval_dataset else ""
        cache_key = f"scaling_agg|metric:{metric}|fpr:{fpr_target}|exp:{exp_prefix}|seeds:{','.join(seeds)}|runs:{','.join(run_labels)}|patterns:{','.join(patterns)}{eval_suffix}"
        if cache_key in cache:
            if verbose:
                print(f"[scaling_agg] Using cached data for {cache_key}", flush=True)
            cached_data = cache[cache_key]
            for idx, label in enumerate(run_labels):
                if label in cached_data:
                    series = cached_data[label]
                    x_vals = series.get('x', [])
                    medians = series.get('medians', [])
                    if x_vals and medians:
                        color = colors[idx]
                        plt.plot(x_vals, medians, 'o-', color=color, label=label, linewidth=2, markersize=6)
            
            # Set labels and styling
            if y_label is None:
                y_label = ("Median AUC across probes" if metric == 'auc' else f"Median Recall at {fpr_target*100}% FPR across probes")
            if x_label is None:
                x_label = "Number of positive examples in the train set"
            
            if eval_dataset:
                plt.title(f"Scaling across Qwen model sizes — OOD ({eval_dataset})", fontsize=14)
            else:
                plt.title("Scaling across Qwen model sizes", fontsize=14)
            plt.ylabel(y_label, fontsize=12)
            plt.xlabel(x_label, fontsize=12)
            plt.xscale('log')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                if verbose:
                    print(f"[scaling_agg] Saving cached figure to {save_path}", flush=True)
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()
            return
    
    # Collect data and compute metrics
    all_run_data = {}
    for idx, (root, label) in enumerate(zip(run_roots, run_labels)):
        if verbose:
            print(f"[scaling_agg] Processing {label}...", flush=True)
        
        exp_dirs = find_experiment_folders(root, seeds[0], exp_prefix)
        if not exp_dirs:
            if verbose:
                print(f"[scaling_agg] No experiment dirs found for {label}", flush=True)
            continue
        exp_dir = exp_dirs[0]
        
        # Collect all scores across all probe patterns for each sample count
        class1_to_all_scores: Dict[int, List[float]] = {}
        
        for pattern_idx, pattern in enumerate(patterns):
            if verbose:
                print(f"[scaling_agg] {label} - pattern {pattern_idx+1}/{len(patterns)}: {pattern}", flush=True)
            
            files = collect_eval_result_files_for_pattern(
                root, seeds, exp_dir.name, evaluation_dirs=None, 
                pattern=pattern, eval_dataset=eval_dataset, require_default=True
            )
            
            if verbose:
                print(f"[scaling_agg] {label} - {pattern}: found {len(files)} files", flush=True)
            
            for file_idx, f in enumerate(files):
                if verbose and file_idx % 10 == 0:
                    print(f"[scaling_agg] {label} - {pattern}: processing file {file_idx+1}/{len(files)}", flush=True)
                
                m = re.search(r'class1_(\d+)', f)
                if not m:
                    continue
                n_pos = int(m.group(1))
                
                # Check cache for this specific file
                file_cache_key = f"file_metric|{f}|{metric}|{fpr_target}"
                if file_cache_key in cache:
                    val = cache[file_cache_key]
                else:
                    try:
                        scores, labels = get_scores_and_labels(f)
                        if metric == 'auc':
                            val = auc_metric(labels, scores)
                        else:
                            val = recall_at_fpr(labels, scores, fpr_target)
                        # Cache the result
                        cache[file_cache_key] = val
                    except Exception as e:
                        if verbose:
                            print(f"[scaling_agg] Error processing {f}: {e}", flush=True)
                        continue
                
                class1_to_all_scores.setdefault(n_pos, []).append(val)
        
        # Compute median across all probes for each sample count
        x_vals = sorted(class1_to_all_scores.keys())
        if not x_vals:
            if verbose:
                print(f"[scaling_agg] No data for {label}", flush=True)
            continue
            
        medians = []
        # Optional detailed accounting for how many values contribute to each median
        if verbose:
            total_vals = sum(len(v) for v in class1_to_all_scores.values())
            print(f"[scaling_agg] {label}: total contributing values across probes = {total_vals}", flush=True)
        for x in x_vals:
            vals = class1_to_all_scores[x]
            if verbose:
                print(f"[scaling_agg] {label}: n_pos={x} count={len(vals)}", flush=True)
            medians.append(np.median(vals))
        
        if verbose:
            print(f"[scaling_agg] {label}: {len(x_vals)} sample counts, {len(medians)} medians", flush=True)
        
        color = colors[idx]
        plt.plot(x_vals, medians, 'o-', color=color, label=label, linewidth=2, markersize=6)
        
        # Store for caching
        all_run_data[label] = {'x': x_vals, 'medians': medians}
    
    # Set labels and styling
    if y_label is None:
        y_label = ("Median AUC across probes" if metric == 'auc' else f"Median Recall at {fpr_target*100}% FPR across probes")
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    
    if eval_dataset:
        plt.title(f"Scaling across Qwen model sizes — OOD ({eval_dataset})", fontsize=14)
    else:
        plt.title("Scaling across Qwen model sizes", fontsize=14)
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        if verbose:
            print(f"[scaling_agg] Saving figure to {save_path}", flush=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        # Save cache
        if cache_dir is not None and cache_key is not None and all_run_data:
            cache[cache_key] = all_run_data
            _save_cache(cache_dir, cache)
            if verbose:
                print(f"[scaling_agg] Saved cache for {cache_key}", flush=True)
    else:
        plt.show()


def plot_llm_upsampling_aggregated_across_probes(
    base_results_dir: Path,
    seeds: List[str],
    exp_prefix: str,
    save_path=None,
    metric: str = 'auc',
    fpr_target: float = 0.01,
    verbose: bool = False,
    eval_dataset: str = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    probe_patterns: Optional[Dict[str, List[str]]] = None,
):
    """
    Generate LLM upsampling plot showing median scores across ALL probe types.
    No error bars or shaded regions - just median lines.
    """
    if seeds is None:
        seeds = ['42']
    
    # Get all probe patterns - use provided patterns or fall back to defaults
    if probe_patterns is None:
        patterns = []
        for plist in default_probe_patterns().values():
            patterns.extend(plist)
    else:
        patterns = []
        for plist in probe_patterns.values():
            patterns.extend(plist)
    # Deduplicate while preserving order
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]
    
    if verbose:
        print(f"[llm_agg] Using {len(patterns)} probe patterns: {patterns}", flush=True)
    
    # Try to load cache
    cache_dir = None
    if save_path is not None:
        cache_dir = Path(save_path).parent
    cache = {}
    cache_key = None
    if cache_dir is not None:
        cache = _load_cache(cache_dir)
        cache_key = f"llm_agg|metric:{metric}|fpr:{fpr_target}|exp:{exp_prefix}|eval:{eval_dataset or 'ALL'}|seeds:{','.join(seeds)}|patterns:{','.join(patterns)}"
        if cache_key in cache:
            if verbose:
                print(f"[llm_agg] Using cached data for {cache_key}", flush=True)
            cached_data = cache[cache_key]
            plt.figure(figsize=(6, 4))
            sorted_factors = sorted(cached_data.keys(), key=int)
            colors = plt.cm.Reds(np.linspace(0.2, 0.98, max(2, len(sorted_factors))))
            
            for idx, factor in enumerate(sorted_factors):
                series = cached_data[factor]
                x_vals = series.get('x', [])
                medians = series.get('medians', [])
                if x_vals and medians:
                    color = colors[idx]
                    label = "no upsampling" if int(factor) == 1 else f"upsample x{factor}"
                    plt.plot(x_vals, medians, 'o-', color=color, label=label, linewidth=2, markersize=6)
            
            # Set labels and styling
            if y_label is None:
                y_label = ("Median AUC across probes" if metric == 'auc' else f"Median Recall at {fpr_target*100}% FPR across probes")
            if x_label is None:
                x_label = "Number of positive examples in the train set"
            
            plt.title("LLM upsampling", fontsize=14)
            plt.ylabel(y_label, fontsize=12)
            plt.xlabel(x_label, fontsize=12)
            plt.xscale('log')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                if verbose:
                    print(f"[llm_agg] Saving cached figure to {save_path}", flush=True)
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()
            return
    
    experiment_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not experiment_dirs:
        print(f"No experiment folders starting with {exp_prefix} found for seed {seeds[0]}")
        return
    exp_dir = experiment_dirs[0]
    
    # Collect all scores across all probe patterns, grouped by upsampling factor and real positives
    from collections import defaultdict
    values_by_factor: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    if verbose:
        print(f"[llm_agg] Processing {len(patterns)} patterns...", flush=True)
    
    for pattern_idx, pattern in enumerate(patterns):
        if verbose:
            print(f"[llm_agg] Pattern {pattern_idx+1}/{len(patterns)}: {pattern}", flush=True)
        
        files = collect_eval_result_files_for_pattern(
            base_results_dir, seeds, exp_dir.name, evaluation_dirs=None,
            pattern=pattern, eval_dataset=eval_dataset, require_default=True
        )
        
        if verbose:
            print(f"[llm_agg] {pattern}: found {len(files)} files", flush=True)
        
        for file_idx, f in enumerate(files):
            if verbose and file_idx % 10 == 0:
                print(f"[llm_agg] {pattern}: processing file {file_idx+1}/{len(files)}", flush=True)
            
            n_real, factor = parse_llm_upsampling_from_filename(f)
            if n_real is None:
                # try to fall back to class1 capture
                m = re.search(r'class1_(\d+)', f)
                if not m:
                    continue
                n_real = int(m.group(1))
                factor = 1
            
            # Check cache for this specific file
            file_cache_key = f"file_metric|{f}|{metric}|{fpr_target}"
            if file_cache_key in cache:
                val = cache[file_cache_key]
            else:
                try:
                    scores, labels = get_scores_and_labels(f)
                    if metric == 'auc':
                        val = auc_metric(labels, scores)
                    else:
                        val = recall_at_fpr(labels, scores, fpr_target)
                    # Cache the result
                    cache[file_cache_key] = val
                except Exception as e:
                    if verbose:
                        print(f"[llm_agg] Error processing {f}: {e}", flush=True)
                    continue
            
            values_by_factor[factor][n_real].append(val)
    
    if not values_by_factor:
        print("No data found for LLM upsampling aggregated plot")
        return
    
    plt.figure(figsize=(6, 4))
    # Sort factors with 1 first, then ascending
    sorted_factors = sorted(values_by_factor.keys())
    colors = plt.cm.Reds(np.linspace(0.2, 0.98, max(2, len(sorted_factors))))
    
    # Store for caching
    all_factor_data = {}
    
    for idx, factor in enumerate(sorted_factors):
        if verbose:
            print(f"[llm_agg] Processing factor {factor}...", flush=True)
        
        inner_map = values_by_factor[factor]
        x_vals = sorted(inner_map.keys())
        if not x_vals:
            continue
        
        # Compute median across all probes for each sample count
        medians = []
        for x in x_vals:
            vals = inner_map[x]
            medians.append(np.median(vals))
        
        if verbose:
            print(f"[llm_agg] factor={factor}: {len(x_vals)} sample counts, {len(medians)} medians", flush=True)
        
        color = colors[idx]
        # Do not use legend labels; color encodes factor
        plt.plot(x_vals, medians, 'o-', color=color, linewidth=2, markersize=6)
        
        # Store for caching
        all_factor_data[str(factor)] = {'x': x_vals, 'medians': medians}
    
    # Set labels and styling
    if y_label is None:
        y_label = ("Median AUC across probes" if metric == 'auc' else f"Median Recall at {fpr_target*100}% FPR across probes")
    if x_label is None:
        x_label = "Number of positive examples in the train set"
    
    plt.title("LLM upsampling", fontsize=14)
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.xscale('log')
    # Replace legend with a colorbar for upsampling factors
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.Reds, norm=mpl.colors.Normalize(vmin=min(sorted_factors), vmax=max(sorted_factors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('LLM upsampling factor', fontsize=10)
    cbar.set_ticks([1, 2, 3, 4, 5, 10, 20])
    cbar.set_ticklabels(['1', '2', '3', '4', '5', '10', '20'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        if verbose:
            print(f"[llm_agg] Saving figure to {save_path}", flush=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        # Save cache
        if cache_dir is not None and cache_key is not None and all_factor_data:
            cache[cache_key] = all_factor_data
            _save_cache(cache_dir, cache)
            if verbose:
                print(f"[llm_agg] Saved cache for {cache_key}", flush=True)
    else:
        plt.show()
