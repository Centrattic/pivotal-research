import json
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def get_scores_and_labels(result_path: str):
    with open(result_path, 'r') as f:
        d = json.load(f)
    if 'scores' in d and isinstance(d['scores'], dict) and 'scores' in d['scores']:
        return np.array(d['scores']['scores']), np.array(d['scores']['labels'])
    if 'filtered_scores' in d:
        return np.array(d['filtered_scores']['scores']), np.array(d['filtered_scores']['labels'])
    if 'all_scores' in d:
        return np.array(d['all_scores']['scores']), np.array(d['all_scores']['labels'])
    if 'scores' in d and isinstance(d['scores'], list):
        return np.array(d['scores']), np.array(d.get('labels', []))
    raise ValueError(f"Could not parse scores/labels from {result_path}")


def find_experiment_folders(results_root: Path, seed: str, exp_prefix: str) -> List[Path]:
    seed_dir = results_root / f"seed_{seed}"
    if not seed_dir.exists():
        return []
    return [seed_dir / sub for sub in os.listdir(seed_dir) if sub.startswith(exp_prefix)]


def extract_eval_and_train_from_filename(path_str: str) -> Tuple[Optional[str], Optional[str]]:
    eval_match = re.search(r'eval_on_([^_]+(?:_[^_]+)*?)__', path_str)
    train_match = re.search(r'train_on_([^_]+(?:_[^_]+)*)_', path_str)
    return (eval_match.group(1) if eval_match else None,
            train_match.group(1) if train_match else None)


def default_probe_patterns() -> Dict[str, List[str]]:
    return {
        'linear': ['linear_last', 'linear_max', 'linear_mean', 'linear_softmax'],
        'sae': ['sae_16k_l0_408', 'sae_262k_l0_259'],
        'attention': ['attention'],
        'act_sim': ['act_sim_last', 'act_sim_max', 'act_sim_mean'],
    }


def is_default_probe_file(filepath: str, probe_type: str) -> bool:
    name = os.path.basename(filepath)
    lowered = name.lower()
    if probe_type in ('linear', 'sae'):
        # For linear and SAE, default C=1.0 is omitted from filenames.
        # If any explicit C is present, treat as non-default.
        if re.search(r'(^|[_-])c\d', lowered):
            return False
        # SAE default includes topk=3584; allow either omitted or explicitly topk3584.
        if probe_type == 'sae':
            m_topk = re.search(r'topk(\d+)', lowered)
            if m_topk and m_topk.group(1) != '3584':
                return False
    elif probe_type == 'attention':
        # For attention, default lr=5e-4 and wd=0.0 are omitted from filenames.
        # If any lr or wd is explicitly present, treat as non-default.
        if re.search(r'(^|[_-])lr[0-9eE\.-]+', lowered) or re.search(r'(^|[_-])wd[0-9eE\.-]+', lowered):
            return False
    return True


def collect_eval_result_files_for_pattern(
    base_results_dir: Path,
    seeds: List[str],
    experiment_folder: str,
    evaluation_dirs: Optional[List[str]],
    pattern: str,
    eval_dataset: Optional[str] = None,
    require_default: bool = True,
) -> List[str]:
    if evaluation_dirs is None:
        evaluation_dirs = ['test_eval', 'gen_eval']
    collected: List[str] = []
    for seed in seeds:
        exp_dir = base_results_dir / f"seed_{seed}" / experiment_folder
        for eval_dir_name in evaluation_dirs:
            eval_dir = exp_dir / eval_dir_name
            if not eval_dir.exists():
                continue
            glob_pat = f"*{pattern}*_results.json"
            files = glob.glob(str(eval_dir / (f"eval_on_{eval_dataset}__{glob_pat}" if eval_dataset else glob_pat)))
            for f in files:
                if 'linear' in pattern:
                    ptype = 'linear'
                elif 'sae' in pattern:
                    ptype = 'sae'
                elif 'attention' in pattern:
                    ptype = 'attention'
                elif 'act_sim' in pattern:
                    ptype = 'act_sim'
                else:
                    ptype = 'unknown'
                if not require_default or is_default_probe_file(f, ptype):
                    collected.append(f)
    return collected


def parse_llm_upsampling_from_filename(path_str: str) -> Tuple[Optional[int], int]:
    """
    Attempt to extract (n_real_positive_samples, upsampling_factor) from filename.
    If no explicit upsampling factor is present, factor defaults to 1.
    """
    # Pattern used in exp3 runs e.g., ... llm_negXXXX_posNN_(K)x ...
    m = re.search(r'pos(\d+)_([1-9]\d*)x', path_str)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None, 1
    # Fallback to class1 count if available
    m2 = re.search(r'class1_(\d+)', path_str)
    if m2:
        try:
            return int(m2.group(1)), 1
        except Exception:
            return None, 1
    return None, 1


def find_inner_results_folder(exp_dir: Path) -> Optional[Path]:
    """Return the most relevant inner results folder under an experiment directory.
    Preference order: 'trained' (legacy), then 'test_eval', then 'gen_eval'.
    """
    for name in ['trained', 'test_eval', 'gen_eval']:
        candidate = exp_dir / name
        if candidate.exists():
            return candidate
    return None


def collect_result_files_for_pattern(
    base_results_dir: Path,
    seeds: List[str],
    experiment_folder: str,
    inner_folder: str,
    pattern: str,
    eval_dataset: Optional[str] = None,
    require_default: bool = True,
) -> List[str]:
    collected: List[str] = []
    for seed in seeds:
        root = base_results_dir / f"seed_{seed}" / experiment_folder / inner_folder
        if not root.exists():
            continue
        glob_pat = f"*{pattern}*_results.json"
        if eval_dataset:
            files = glob.glob(str(root / f"eval_on_{eval_dataset}__{glob_pat}"))
        else:
            files = glob.glob(str(root / glob_pat))
        for f in files:
            if 'linear' in pattern:
                ptype = 'linear'
            elif 'sae' in pattern:
                ptype = 'sae'
            elif 'attention' in pattern:
                ptype = 'attention'
            elif 'act_sim' in pattern:
                ptype = 'act_sim'
            else:
                ptype = 'unknown'
            if not require_default or is_default_probe_file(f, ptype):
                collected.append(f)
    return collected


def _get_scores_and_labels_from_result_file(result_path: str):
    return get_scores_and_labels(result_path)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    from scipy.special import expit
    return float(roc_auc_score(labels, expit(scores)))


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    from scipy.special import expit
    s = expit(scores)
    thresholds = np.unique(s)[::-1]
    best = 0.0
    for thresh in thresholds:
        preds = (s >= thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if fpr <= fpr_target and rec > best:
            best = rec
    return float(best)


def draw_iqr_line(ax, x_vals, medians, q25, q75, label, color):
    ax.plot(x_vals, medians, 'o-', label=label, linewidth=2, color=color, markersize=6)
    if any(a != b for a, b in zip(q25, q75)):
        ax.fill_between(x_vals, q25, q75, alpha=0.2, color=color)


def autoset_ylim_from_bands(ax, all_lower: List[float], pad: float = 0.05):
    if all_lower:
        y_min = max(0.0, min(all_lower) - pad)
        ax.set_ylim(y_min, 1.0)


