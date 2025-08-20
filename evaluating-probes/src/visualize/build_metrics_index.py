import argparse
from pathlib import Path
import csv
from typing import List

# Reuse robust readers/metrics from viz_core
from .viz_core import get_scores_and_labels, auc as auc_metric, recall_at_fpr


def enumerate_result_files(results_root: Path, run_name: str) -> List[Path]:
    run_root = results_root / run_name
    if not run_root.exists():
        return []
    files: List[Path] = []
    # seed_* / <exp_dir> / {gen_eval,test_eval,val_eval} / *_results.json
    for seed_dir in sorted(run_root.glob('seed_*')):
        if not seed_dir.is_dir():
            continue
        for exp_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            for eval_name in ('gen_eval', 'test_eval', 'val_eval'):
                eval_dir = exp_dir / eval_name
                if not eval_dir.exists():
                    continue
                files.extend(sorted(eval_dir.glob('*_results.json')))
    return files


def main():
    parser = argparse.ArgumentParser(description='Index AUC and Recall@FPR for all results JSONs under specified run_names.')
    parser.add_argument('-r', '--run-names', nargs='+', required=True, help='One or more run_name folders under results/.')
    parser.add_argument('--fpr', type=float, default=0.01, help='Target FPR for recall metric (default: 0.01).')
    parser.add_argument('-o', '--output', default='src/visualizations/metrics_index.csv', help='Output CSV path.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging.')
    args = parser.parse_args()

    # Resolve repository root as src/.. (this file lives at src/visualize/)
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / 'results'

    # Collect files from all run_names
    all_files: List[Path] = []
    for rn in args.run_names:
        files = enumerate_result_files(results_root, rn)
        if args.verbose:
            print(f"[index] {rn}: found {len(files)} result files")
        all_files.extend(files)

    if not all_files:
        if args.verbose:
            print('[index] No result files found. Nothing to write.')
        # Still create an empty CSV with header
        out_path = repo_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'recall', 'auc'])
            writer.writeheader()
        return

    rows = []
    for path in all_files:
        try:
            scores, labels = get_scores_and_labels(str(path))
            auc_val = auc_metric(labels, scores)
            rec_val = recall_at_fpr(labels, scores, args.fpr)
            # Filename must start with results/...
            rel = path.relative_to(repo_root)
            rows.append({'filename': str(rel), 'recall': f"{rec_val:.6f}", 'auc': f"{auc_val:.6f}"})
        except Exception as e:
            if args.verbose:
                print(f"[index] Failed on {path}: {e}")
            continue

    # Write CSV
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'recall', 'auc'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.verbose:
        print(f"[index] Wrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()


