import json
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import textwrap

def get_scores_and_labels(result_path: str):
    """Return (scores, labels) as numpy arrays from a result json.

    Supports several schema variants. Raises ValueError if neither scores nor a
    reasonable surrogate (e.g., probabilities/logits) can be found.
    """
    with open(result_path, 'r') as f:
        d = json.load(f)

    # Common nested structure
    if 'scores' in d and isinstance(d['scores'], dict):
        inner = d['scores']
        if 'scores' in inner and 'labels' in inner:
            return np.array(inner['scores']), np.array(inner['labels'])

    # Alternate top-level containers
    for key in ['filtered_scores', 'all_scores']:
        if key in d and isinstance(d[key], dict) and 'scores' in d[key] and 'labels' in d[key]:
            return np.array(d[key]['scores']), np.array(d[key]['labels'])

    # Flat list-style
    if isinstance(d.get('scores'), list):
        labels = d.get('labels', [])
        return np.array(d['scores']), np.array(labels)

    # Other common naming conventions
    for score_key in ['logits', 'probabilities', 'probs', 'y_score', 'y_prob', 'y_scores']:
        if score_key in d:
            # labels keys commonly used
            for label_key in ['labels', 'y_true', 'targets', 'y']:
                if label_key in d:
                    return np.array(d[score_key]), np.array(d[label_key])

    # Do NOT fall back to hard predictions for ROC/Recall metrics; require real-valued scores

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
    """Canonical default patterns used when config-specific patterns are unavailable."""
    return {
        'linear': [
            'linear_last', 'linear_max', 'linear_mean', 'linear_softmax',
            'sklearn_linear_last', 'sklearn_linear_max', 'sklearn_linear_mean', 'sklearn_linear_softmax',
        ],
        'sae': [
            # Gemma lens presets
            'sae_16k_l0_408', 'sae_262k_l0_259',
            # Generic SAE aggregations (model-agnostic)
            'sae_mean', 'sae_last', 'sae_max', 'sae_softmax',
            # LLaMA 3.3 70B variants
            'sae_llama33b_mean', 'sae_llama33b_last', 'sae_llama33b_max', 'sae_llama33b_softmax',
            # Qwen3 family variants
            'sae_qwen3_0.6b_last', 'sae_qwen3_0.6b_mean', 'sae_qwen3_0.6b_max', 'sae_qwen3_0.6b_softmax',
            'sae_qwen3_1.7b_last', 'sae_qwen3_1.7b_mean', 'sae_qwen3_1.7b_max', 'sae_qwen3_1.7b_softmax',
            'sae_qwen3_4b_last', 'sae_qwen3_4b_mean', 'sae_qwen3_4b_max', 'sae_qwen3_4b_softmax',
            'sae_qwen3_8b_last', 'sae_qwen3_8b_mean', 'sae_qwen3_8b_max', 'sae_qwen3_8b_softmax',
            'sae_qwen3_14b_last', 'sae_qwen3_14b_mean', 'sae_qwen3_14b_max', 'sae_qwen3_14b_softmax',
        ],
        'attention': ['attention'],
        'act_sim': ['act_sim_last', 'act_sim_max', 'act_sim_mean'],
    }


def patterns_from_config(architectures: List[Dict]) -> Dict[str, List[str]]:
    """Build probe pattern groups strictly from architecture config_names in the config.

    Returns mapping: probe_type -> list of pattern substrings to match in filenames.
    """
    groups: Dict[str, List[str]] = {'linear': [], 'sae': [], 'attention': [], 'act_sim': []}
    for arch in architectures:
        config_name = arch.get('config_name') or arch.get('name') or ''
        lname = config_name.lower()
        if 'sae' in lname:
            groups['sae'].append(config_name)
        elif 'act_sim' in lname:
            groups['act_sim'].append(config_name)
        elif 'attention' in lname:
            groups['attention'].append(config_name)
        elif 'linear' in lname:
            groups['linear'].append(config_name)
    # Deduplicate
    for k in groups:
        seen = set()
        groups[k] = [p for p in groups[k] if not (p in seen or seen.add(p))]
    return groups


def labels_from_config(architectures: List[Dict]) -> Dict[str, str]:
    """Map architecture names to publication-ready labels."""
    labels: Dict[str, str] = {}
    for arch in architectures:
        # Prefer config_name (captures aggregation/details), then fall back to name
        cfg_name = arch.get('config_name') or arch.get('name') or ''
        labels[cfg_name] = paper_label_for_probe(cfg_name)
    return labels

def is_default_probe_file(filepath: str, probe_type: str) -> bool:
    name = os.path.basename(filepath)
    lowered = name.lower()
    if probe_type in ('linear', 'sae'):
        # For linear and SAE, default C=1.0 is allowed whether omitted or explicitly present.
        m_c = re.search(r'(^|[_-])c([0-9eE\.+-]+)', lowered)
        if m_c:
            try:
                c_val = float(m_c.group(2))
                if abs(c_val - 1.0) > 1e-9:
                    return False
            except Exception:
                # Unknown C value format; treat as non-default to be safe
                return False
        # SAE default includes topk=3584; allow either omitted or explicitly topk3584.
        if probe_type == 'sae':
            m_topk = re.search(r'topk(\d+)', lowered)
            if m_topk and m_topk.group(1) != '3584':
                return False
    elif probe_type == 'attention':
        # For attention, treat lr=5e-4 and wd=0.0 as defaults (allowed when explicitly present).
        m_lr = re.search(r'(^|[_-])lr([0-9eE\.+-]+)', lowered)
        m_wd = re.search(r'(^|[_-])wd([0-9eE\.+-]+)', lowered)
        try:
            if m_lr:
                lr_val = float(m_lr.group(2))
                if abs(lr_val - 5e-4) > 1e-12:
                    return False
            if m_wd:
                wd_val = float(m_wd.group(2))
                if abs(wd_val - 0.0) > 1e-12:
                    return False
        except Exception:
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
    # If scores already look like probabilities in [0, 1], use as-is; otherwise apply sigmoid
    try:
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        scores_prob = scores if (s_min >= 0.0 and s_max <= 1.0) else expit(scores)
    except Exception:
        scores_prob = expit(scores)
    return float(roc_auc_score(labels, scores_prob))

def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    from scipy.special import expit
    # Use probabilities directly if scores in [0,1], otherwise apply sigmoid
    try:
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        s = scores if (s_min >= 0.0 and s_max <= 1.0) else expit(scores)
    except Exception:
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

def compute_metric_from_result(result_path: str, metric: str, fpr_target: float) -> Optional[float]:
    """Compute a metric from a results json, even if only raw scores exist.

    - metric: 'auc' or 'recall' (recall@FPR)
    Returns None if computation fails.
    """
    try:
        scores, labels = get_scores_and_labels(result_path)
    except Exception:
        return None
    try:
        if metric == 'auc':
            return auc(labels, scores)
        else:
            return recall_at_fpr(labels, scores, fpr_target)
    except Exception:
        return None

def draw_iqr_line(ax, x_vals, medians, q25, q75, label, color):
    # Lines with small dot markers, slightly thinner line width
    ax.plot(x_vals, medians, 'o-', label=label, linewidth=1.5, color=color, markersize=4)
    if any(a != b for a, b in zip(q25, q75)):
        ax.fill_between(x_vals, q25, q75, alpha=0.2, color=color)

def autoset_ylim_from_bands(ax, all_lower: List[float], pad: float = 0.05):
    if all_lower:
        y_min = max(0.0, min(all_lower) - pad)
        ax.set_ylim(y_min, 1.0)

# Human-readable labels for paper figures
_PAPER_LABEL_MAP: Dict[str, str] = {
    'linear_last': 'Linear (last token)',
    'linear_max': 'Linear (max token)',
    'linear_mean': 'Linear (mean pool)',
    'linear_softmax': 'Linear (softmax pool)',
    'sklearn_linear_last': 'Linear (last token)',
    'sklearn_linear_max': 'Linear (max token)',
    'sklearn_linear_mean': 'Linear (mean pool)',
    'sklearn_linear_softmax': 'Linear (softmax pool)',
    'attention': 'Attention',
    'act_sim_last': 'Cosine similarity (last token)',
    'act_sim_max': 'Cosine similarity (max token)',
    'act_sim_mean': 'Cosine similarity (mean pool)',
    'act_sim_softmax': 'Cosine similarity (softmax pool)',
    # SAE Lens (Gemma) presets
    'sae_16k_l0_408_last': 'SAE (width=16k, L0=408, last token)',
    'sae_16k_l0_408_mean': 'SAE (width=16k, L0=408, mean pool)',
    'sae_16k_l0_408_max': 'SAE (width=16k, L0=408, max token)',
    'sae_16k_l0_408_softmax': 'SAE (width=16k, L0=408, softmax pool)',
    'sae_262k_l0_259_last': 'SAE (width=262k, L0=259, last token)',
    'sae_262k_l0_259_mean': 'SAE (width=262k, L0=259, mean pool)',
    'sae_262k_l0_259_max': 'SAE (width=262k, L0=259, max token)',
    'sae_262k_l0_259_softmax': 'SAE (width=262k, L0=259, softmax pool)',
    # SAE LLaMA 3.3 70B
    'sae_llama33b_last': 'SAE (Llama-3.3-70B, last token)',
    'sae_llama33b_mean': 'SAE (Llama-3.3-70B, mean pool)',
    'sae_llama33b_max': 'SAE (Llama-3.3-70B, max pool)',
    'sae_llama33b_softmax': 'SAE (Llama-3.3-70B, softmax)',
    # Qwen3 family
    'sae_qwen3_0.6b_last': 'SAE (Qwen3 0.6B, last token)',
    'sae_qwen3_0.6b_mean': 'SAE (Qwen3 0.6B, mean pool)',
    'sae_qwen3_0.6b_max': 'SAE (Qwen3 0.6B, max pool)',
    'sae_qwen3_0.6b_softmax': 'SAE (Qwen3 0.6B, softmax)',
    'sae_qwen3_1.7b_last': 'SAE (Qwen3 1.7B, last token)',
    'sae_qwen3_1.7b_mean': 'SAE (Qwen3 1.7B, mean pool)',
    'sae_qwen3_1.7b_max': 'SAE (Qwen3 1.7B, max pool)',
    'sae_qwen3_1.7b_softmax': 'SAE (Qwen3 1.7B, softmax)',
    'sae_qwen3_4b_last': 'SAE (Qwen3 4B, last token)',
    'sae_qwen3_4b_mean': 'SAE (Qwen3 4B, mean pool)',
    'sae_qwen3_4b_max': 'SAE (Qwen3 4B, max pool)',
    'sae_qwen3_4b_softmax': 'SAE (Qwen3 4B, softmax)',
    'sae_qwen3_8b_last': 'SAE (Qwen3 8B, last token)',
    'sae_qwen3_8b_mean': 'SAE (Qwen3 8B, mean pool)',
    'sae_qwen3_8b_max': 'SAE (Qwen3 8B, max pool)',
    'sae_qwen3_8b_softmax': 'SAE (Qwen3 8B, softmax)',
    'sae_qwen3_14b_last': 'SAE (Qwen3 14B, last token)',
    'sae_qwen3_14b_mean': 'SAE (Qwen3 14B, mean pool)',
    'sae_qwen3_14b_max': 'SAE (Qwen3 14B, max pool)',
    'sae_qwen3_14b_softmax': 'SAE (Qwen3 14B, softmax)',
    # Mass-mean
    'mass_mean': 'Mass Mean',
    'default_mass_mean': 'Mass Mean',
    'mass_mean_iid': 'Mass Mean (IID)',
    # Defaults aliases
    'default_sklearn_linear': 'Error',
    'default_act_sim': 'Error',
}

def paper_label_for_probe(probe_name: str) -> str:
    # First, attempt an exact mapping from known config names
    if probe_name in _PAPER_LABEL_MAP:
        return _PAPER_LABEL_MAP[probe_name]
    lowered = probe_name.lower()
    # Linear variants
    if 'linear_softmax' in lowered:
        return 'Linear (softmax)'
    if 'linear_mean' in lowered:
        return 'Linear (mean pool)'
    if 'linear_max' in lowered:
        return 'Linear (max pool)'
    if 'linear_last' in lowered:
        return 'Linear (last token)'
    # Activation similarity
    if 'act_sim_mean' in lowered:
        return 'Activation Similarity (mean)'
    if 'act_sim_max' in lowered:
        return 'Activation Similarity (max)'
    if 'act_sim_last' in lowered:
        return 'Activation Similarity (last)'
    # Attention
    if 'attention' in lowered:
        return 'Attention'
    # SAE with width/L0 if present
    if 'sae' in lowered:
        pool = None
        if 'sae_mean' in lowered:
            pool = 'mean pool'
        elif 'sae_max' in lowered:
            pool = 'max pool'
        elif 'sae_last' in lowered:
            pool = 'last token'
        # Try to extract width and L0
        m = re.search(r'sae[_-](\d+\s*[kK])?_?l0[_-](\d+)', lowered)
        if m:
            width = m.group(1)
            l0 = m.group(2)
            width_str = f"width={width}" if width else None
            parts = [p for p in [width_str, f"L0={l0}", pool] if p]
            return f"SAE ({', '.join(parts)})" if parts else 'SAE'
        return f"SAE ({pool})" if pool else 'SAE'
    return probe_name

def map_probe_labels(probe_names: List[str]) -> Dict[str, str]:
    return {name: paper_label_for_probe(name) for name in probe_names}


def wrap_text_for_plot(text: Optional[str], max_line_length: int = 36) -> Optional[str]:
    """Insert newlines into text to avoid long single-line labels in figures.

    If text is None or already short, returns as-is. Breaks on word boundaries.
    """
    if text is None:
        return None
    if len(text) <= max_line_length:
        return text
    return "\n".join(textwrap.wrap(text, width=max_line_length))