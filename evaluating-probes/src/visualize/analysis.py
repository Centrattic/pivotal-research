import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.visualize.utils_viz import (
    find_experiment_folders,
    find_inner_results_folder,
    collect_result_files_for_pattern,
    default_probe_patterns,
    _get_scores_and_labels_from_result_file,
    is_default_probe_file,
)


def _compute_auc_and_recall_arrays(files: List[str], fpr_target: float = 0.01) -> Tuple[List[float], List[float]]:
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score

    aucs: List[float] = []
    recalls: List[float] = []
    for f in files:
        try:
            scores, labels = _get_scores_and_labels_from_result_file(f)
            sig = expit(scores)
            # AUC
            try:
                aucs.append(float(roc_auc_score(labels, sig)))
            except Exception:
                continue
            # Recall@FPR
            thresholds = np.unique(sig)[::-1]
            best_recall = 0.0
            for t in thresholds:
                preds = (sig >= t).astype(int)
                tp = int(np.sum((preds == 1) & (labels == 1)))
                fn = int(np.sum((preds == 0) & (labels == 1)))
                fp = int(np.sum((preds == 1) & (labels == 0)))
                tn = int(np.sum((preds == 0) & (labels == 0)))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if fpr <= fpr_target and recall > best_recall:
                    best_recall = recall
            recalls.append(float(best_recall))
        except Exception:
            continue
    return aucs, recalls


def _enumerate_eval_datasets(base_results_dir: Path, seeds: List[str], exp_prefix: str) -> List[str]:
    """Look into files to find unique eval_on datasets for the given experiment prefix."""
    datasets: set = set()
    exp_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not exp_dirs:
        return []
    exp_dir = exp_dirs[0]
    inner = find_inner_results_folder(exp_dir)
    if inner is None:
        return []
    for root, _dirs, files in os.walk(inner):
        for fn in files:
            if not fn.endswith('_results.json'):
                continue
            m = re.search(r'eval_on_([^_]+(?:_[^_]+)*)__', fn)
            if m:
                datasets.add(m.group(1))
    return sorted(list(datasets))


def write_best_default_probe_tables(
    results_dir: Path,
    viz_root: Path,
    seeds: List[str],
    exp_prefixes: List[str],
    fpr_target: float = 0.01,
) -> None:
    """For each experiment prefix and each eval dataset, pick the best default probe per type and write CSV summary.

    CSV columns: probe_type, best_pattern, median_auc, median_recall_at_fpr, n_files
    """
    patterns = default_probe_patterns()
    viz_root.mkdir(parents=True, exist_ok=True)

    for exp_prefix in exp_prefixes:
        exp_dirs = find_experiment_folders(results_dir, seeds[0], exp_prefix)
        if not exp_dirs:
            continue
        exp_dir = exp_dirs[0]
        inner = find_inner_results_folder(exp_dir)
        if inner is None:
            continue

        eval_datasets = _enumerate_eval_datasets(results_dir, seeds, exp_prefix)
        if not eval_datasets:
            # fall back to None to aggregate across all evals
            eval_datasets = [None]

        for eval_dataset in eval_datasets:
            rows = []
            for ptype, pats in patterns.items():
                med_auc_by_pattern: Dict[str, float] = {}
                med_rec_by_pattern: Dict[str, float] = {}
                n_by_pattern: Dict[str, int] = {}
                for pattern in pats:
                    files = collect_result_files_for_pattern(
                        results_dir,
                        seeds,
                        exp_dir.name,
                        inner.name,
                        pattern,
                        eval_dataset=eval_dataset,
                        require_default=True,
                    )
                    aucs, recs = _compute_auc_and_recall_arrays(files, fpr_target=fpr_target)
                    if aucs:
                        med_auc_by_pattern[pattern] = float(np.median(aucs))
                        med_rec_by_pattern[pattern] = float(np.median(recs)) if recs else float('nan')
                        n_by_pattern[pattern] = len(aucs)
                if med_auc_by_pattern:
                    best_pattern = max(med_auc_by_pattern.items(), key=lambda kv: kv[1])[0]
                    rows.append({
                        'probe_type': ptype,
                        'best_pattern': best_pattern,
                        'median_auc': f"{med_auc_by_pattern[best_pattern]:.4f}",
                        'median_recall_at_fpr': f"{med_rec_by_pattern.get(best_pattern, float('nan')):.4f}",
                        'n_files': n_by_pattern.get(best_pattern, 0),
                        'eval_dataset': eval_dataset or 'ALL',
                        'experiment': exp_dir.name,
                    })

            if rows:
                csv_name = f"best_default_probes_{exp_dir.name}_{eval_dataset if eval_dataset else 'ALL'}.csv"
                out_path = viz_root / csv_name
                with open(out_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Wrote {out_path}")


def _discover_exp_and_inner(base_results_dir: Path, seeds: List[str], exp_prefix: str) -> tuple[Optional[str], Optional[str]]:
    exp_dirs = find_experiment_folders(base_results_dir, seeds[0], exp_prefix)
    if not exp_dirs:
        return None, None
    exp_dir = exp_dirs[0]
    inner = find_inner_results_folder(exp_dir)
    if inner is None:
        return exp_dir.name, None
    return exp_dir.name, inner.name


def _probe_type_from_pattern(pattern: str) -> str:
    if 'linear' in pattern:
        return 'linear'
    if 'sae' in pattern:
        return 'sae'
    if 'attention' in pattern:
        return 'attention'
    if 'act_sim' in pattern:
        return 'act_sim'
    return 'unknown'


def _aggregate_metric(files: List[str], metric: str, fpr_target: float) -> List[float]:
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    values: List[float] = []
    for f in files:
        try:
            scores, labels = _get_scores_and_labels_from_result_file(f)
            if metric == 'auc':
                values.append(float(roc_auc_score(labels, expit(scores))))
            else:
                # recall@fpr
                sig = expit(scores)
                thresholds = np.unique(sig)[::-1]
                best = 0.0
                for t in thresholds:
                    preds = (sig >= t).astype(int)
                    tp = int(np.sum((preds == 1) & (labels == 1)))
                    fn = int(np.sum((preds == 0) & (labels == 1)))
                    fp = int(np.sum((preds == 1) & (labels == 0)))
                    tn = int(np.sum((preds == 0) & (labels == 0)))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    if fpr <= fpr_target and rec > best:
                        best = rec
                values.append(float(best))
        except Exception:
            continue
    return values


def _collect_exp2_or_4_metrics(
    base_results_dir: Path,
    seeds: List[str],
    exp_prefix: str,
    pattern: str,
    eval_dataset: Optional[str],
    metric: str,
    fpr_target: float,
) -> Dict[int, float]:
    exp_folder, inner_name = _discover_exp_and_inner(base_results_dir, seeds, exp_prefix)
    if exp_folder is None or inner_name is None:
        return {}
    files = collect_result_files_for_pattern(
        base_results_dir,
        seeds,
        exp_folder,
        inner_name,
        pattern,
        eval_dataset=eval_dataset,
        require_default=True,
    )
    # group by class1 count
    buckets: Dict[int, List[str]] = {}
    for f in files:
        m = re.search(r'class1_(\d+)', f)
        if not m:
            continue
        n = int(m.group(1))
        buckets.setdefault(n, []).append(f)
    result: Dict[int, float] = {}
    for n, group in buckets.items():
        vals = _aggregate_metric(group, metric=metric, fpr_target=fpr_target)
        if vals:
            result[n] = float(np.median(vals))
    return result


def _collect_exp3_best_over_upsampling(
    base_results_dir: Path,
    seeds: List[str],
    pattern: str,
    eval_dataset: Optional[str],
    metric: str,
    fpr_target: float,
) -> Dict[int, float]:
    """Return mapping of real positive samples -> best metric across upsampling factors for default probe."""
    exp_folder, inner_name = _discover_exp_and_inner(base_results_dir, seeds, '3-')
    if exp_folder is None or inner_name is None:
        return {}
    # collect files for pattern
    ptype = _probe_type_from_pattern(pattern)
    collected: List[str] = []
    for seed in seeds:
        seed_dir = base_results_dir / f'seed_{seed}' / exp_folder / inner_name
        if not seed_dir.exists():
            continue
        if eval_dataset:
            candidates = list(seed_dir.glob(f"eval_on_{eval_dataset}__*{pattern}*_results.json"))
        else:
            candidates = list(seed_dir.glob(f"*{pattern}*_results.json"))
        for f in candidates:
            if is_default_probe_file(str(f), ptype):
                collected.append(str(f))
    # group by (n_real, upsampling_factor)
    from collections import defaultdict
    vals_by_pair: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for f in collected:
        m = re.search(r'llm_neg\d+_pos(\d+)_(\d+)x', f)
        if not m:
            # fallback to class1 capture
            m2 = re.search(r'class1_(\d+)', f)
            if not m2:
                continue
            n_real = int(m2.group(1))
            factor = 1
        else:
            n_real = int(m.group(1))
            factor = int(m.group(2))
        vals_by_pair[(n_real, factor)].append(f)
    # compute median per pair, then best across factor per n_real
    best_by_n: Dict[int, float] = {}
    by_n_factor: Dict[int, List[float]] = {}
    for (n_real, factor), group in vals_by_pair.items():
        vals = _aggregate_metric(group, metric=metric, fpr_target=fpr_target)
        if not vals:
            continue
        med = float(np.median(vals))
        by_n_factor.setdefault(n_real, []).append(med)
    for n_real, arr in by_n_factor.items():
        best_by_n[n_real] = float(np.max(arr))
    return best_by_n


def _equivalent_boost_factor(v_target: float, baseline: Dict[int, float]) -> Optional[float]:
    # find smallest m such that baseline[m] >= v_target
    if not baseline:
        return None
    candidates = sorted(baseline.items())
    for m, v in candidates:
        if v >= v_target:
            return float(m)
    # if never reaches, return None
    return None


def write_boost_and_best_tables(
    results_dir: Path,
    viz_root: Path,
    seeds: List[str],
    fpr_target: float = 0.01,
) -> None:
    """Compute requested boost metrics and best-probe tables and save to a txt file in viz_root."""
    viz_root.mkdir(parents=True, exist_ok=True)
    out_path = viz_root / 'analysis_summary.txt'
    patterns = default_probe_patterns()
    probe_patterns = patterns['linear'] + patterns['sae'] + patterns['attention'] + patterns['act_sim']

    # enumerate eval datasets (use exp2 as reference)
    eval_datasets = _enumerate_eval_datasets(results_dir, seeds, '2-')
    if not eval_datasets:
        eval_datasets = [None]

    lines: List[str] = []
    for metric in ['auc', 'recall_at_fpr']:
        lines.append(f"=== Metric: {metric} ===")
        # Boost vs exp2 (imbalanced) and vs exp4 (balanced)
        per_probe_boost_vs_exp2: Dict[str, List[float]] = {}
        per_probe_boost_vs_exp4: Dict[str, List[float]] = {}

        for eval_dataset in eval_datasets:
            lines.append(f"-- Eval dataset: {eval_dataset or 'ALL'} --")
            # baselines
            baseline2_by_probe: Dict[str, Dict[int, float]] = {}
            baseline4_by_probe: Dict[str, Dict[int, float]] = {}
            best3_by_probe: Dict[str, Dict[int, float]] = {}
            for pattern in probe_patterns:
                b2 = _collect_exp2_or_4_metrics(results_dir, seeds, '2-', pattern, eval_dataset, metric, fpr_target)
                b4 = _collect_exp2_or_4_metrics(results_dir, seeds, '4-', pattern, eval_dataset, metric, fpr_target)
                b3 = _collect_exp3_best_over_upsampling(results_dir, seeds, pattern, eval_dataset, metric, fpr_target)
                if b2:
                    baseline2_by_probe[pattern] = b2
                if b4:
                    baseline4_by_probe[pattern] = b4
                if b3:
                    best3_by_probe[pattern] = b3

            # compute boosts per probe
            for pattern, best_map in best3_by_probe.items():
                boosts2: List[float] = []
                boosts4: List[float] = []
                # iterate over available n in exp3
                for n, v_llm in best_map.items():
                    # vs exp2
                    eq_m2 = _equivalent_boost_factor(v_llm, baseline2_by_probe.get(pattern, {}))
                    if eq_m2 is not None and n > 0:
                        boosts2.append(eq_m2 / n)
                    # vs exp4
                    eq_m4 = _equivalent_boost_factor(v_llm, baseline4_by_probe.get(pattern, {}))
                    if eq_m4 is not None and n > 0:
                        boosts4.append(eq_m4 / n)
                if boosts2:
                    per_probe_boost_vs_exp2.setdefault(pattern, []).extend(boosts2)
                if boosts4:
                    per_probe_boost_vs_exp4.setdefault(pattern, []).extend(boosts4)

        # Summarize boosts
        lines.append("Boost factors (LLM upsampling equivalent real-sample multiplier):")
        for name, container in [("vs_imbalanced(exp2)", per_probe_boost_vs_exp2), ("vs_balanced(exp4)", per_probe_boost_vs_exp4)]:
            # per probe stats
            per_probe_stats: List[Tuple[str, float, float]] = []
            for pattern, arr in container.items():
                if not arr:
                    continue
                per_probe_stats.append((pattern, float(np.mean(arr)), float(np.max(arr))))
            # overall
            avg_over_probes = float(np.mean([t[1] for t in per_probe_stats])) if per_probe_stats else float('nan')
            max_over_probes = float(np.max([t[2] for t in per_probe_stats])) if per_probe_stats else float('nan')
            lines.append(f"  {name}: avg_over_probes={avg_over_probes:.3f}, max_over_probes={max_over_probes:.3f}")
            for pattern, avg_boost, max_boost in sorted(per_probe_stats, key=lambda x: x[0]):
                lines.append(f"    {pattern}: avg={avg_boost:.3f}, max={max_boost:.3f}")

        # Best probe table per n and condition (per-eval dataset)
        lines.append("Best probe per n and condition (per-eval dataset):")
        for eval_dataset in eval_datasets:
            lines.append(f"-- Eval dataset: {eval_dataset or 'ALL'} --")
            # collect per-probe maps again
            per_probe_exp2 = {p: _collect_exp2_or_4_metrics(results_dir, seeds, '2-', p, eval_dataset, metric, fpr_target) for p in probe_patterns}
            per_probe_exp4 = {p: _collect_exp2_or_4_metrics(results_dir, seeds, '4-', p, eval_dataset, metric, fpr_target) for p in probe_patterns}
            per_probe_exp3_best = {p: _collect_exp3_best_over_upsampling(results_dir, seeds, p, eval_dataset, metric, fpr_target) for p in probe_patterns}
            # all ns
            ns = sorted(set().union(*[set(d.keys()) for d in per_probe_exp2.values()] + [set(d.keys()) for d in per_probe_exp4.values()] + [set(d.keys()) for d in per_probe_exp3_best.values()]))
            lines.append("n_samples\tbest_imbalanced(probe,val)\tbest_balanced(probe,val)\tbest_llm_infinite(probe,val)")
            for n in ns:
                # imbalanced
                best_imb = max(((p, per_probe_exp2[p].get(n)) for p in probe_patterns if n in per_probe_exp2[p]), key=lambda x: x[1] if x[1] is not None else -1, default=(None, None))
                # balanced
                best_bal = max(((p, per_probe_exp4[p].get(n)) for p in probe_patterns if n in per_probe_exp4[p]), key=lambda x: x[1] if x[1] is not None else -1, default=(None, None))
                # infinite upsampling
                best_llm = max(((p, per_probe_exp3_best[p].get(n)) for p in probe_patterns if n in per_probe_exp3_best[p]), key=lambda x: x[1] if x[1] is not None else -1, default=(None, None))
                def fmt(t):
                    return f"{t[0]}:{t[1]:.3f}" if t[0] is not None and t[1] is not None else "NA"
                lines.append(f"{n}\t{fmt(best_imb)}\t{fmt(best_bal)}\t{fmt(best_llm)}")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


