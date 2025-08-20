import argparse
from pathlib import Path
import csv
from typing import List, Set
from tqdm import tqdm
from joblib import Parallel, delayed

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

def load_existing_files(csv_path: Path, repo_root: Path) -> Set[str]:
    """Load filenames that already exist in the CSV."""
    if not csv_path.exists():
        return set()
    
    existing_files = set()
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_files.add(row['filename'])
    except Exception as e:
        print(f"[index] Warning: Could not read existing CSV {csv_path}: {e}")
        return set()
    
    return existing_files

def process_file(path: Path, repo_root: Path, fpr: float, verbose: bool):
    """Process a single result file and return metrics."""
    try:
        scores, labels = get_scores_and_labels(str(path))
        auc_val = auc_metric(labels, scores)
        rec_val = recall_at_fpr(labels, scores, fpr)
        # Filename must start with results/...
        rel = path.relative_to(repo_root)
        return {'filename': str(rel), 'recall': f"{rec_val:.6f}", 'auc': f"{auc_val:.6f}"}
    except Exception as e:
        if verbose:
            print(f"[index] Failed on {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Index AUC and Recall@FPR for all results JSONs under specified run_names.')
    parser.add_argument('-r', '--run-names', nargs='+', required=True, help='One or more run_name folders under results/.')
    parser.add_argument('--fpr', type=float, default=0.01, help='Target FPR for recall metric (default: 0.01).')
    parser.add_argument('-o', '--output', default='src/visualizations/metrics_index.csv', help='Output CSV path.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging.')
    parser.add_argument('-j', '--jobs', type=int, default=-1, help='Number of parallel jobs (default: -1 for all cores).')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of all files, ignoring existing CSV.')
    args = parser.parse_args()

    # Resolve repository root as src/.. (this file lives at src/visualize/)
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / 'results'
    out_path = repo_root / args.output

    # Load existing files from CSV if it exists and not forcing
    existing_files = set()
    if not args.force and out_path.exists():
        existing_files = load_existing_files(out_path, repo_root)
        if args.verbose:
            print(f"[index] Found {len(existing_files)} existing entries in {out_path}")

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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'recall', 'auc'])
            writer.writeheader()
        return

    # Filter out files that already exist in CSV
    files_to_process = []
    for path in all_files:
        rel_path = str(path.relative_to(repo_root))
        if rel_path not in existing_files:
            files_to_process.append(path)
    
    if args.verbose:
        print(f"[index] {len(files_to_process)} files need processing (skipping {len(all_files) - len(files_to_process)} existing)")

    if not files_to_process:
        print("[index] All files already processed. Use --force to reprocess everything.")
        return

    # Process files in parallel
    print(f"[index] Processing {len(files_to_process)} files using {args.jobs} parallel jobs...")
    results = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(process_file)(path, repo_root, args.fpr, args.verbose)
        for path in tqdm(files_to_process, desc="Processing files")
    )
    
    # Filter out None results (failed files)
    new_rows = [r for r in results if r is not None]

    # Load existing rows if CSV exists
    existing_rows = []
    if out_path.exists() and not args.force:
        try:
            with open(out_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        except Exception as e:
            print(f"[index] Warning: Could not read existing CSV, starting fresh: {e}")
            existing_rows = []

    # Combine existing and new rows
    all_rows = existing_rows + new_rows

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'recall', 'auc'])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    if args.verbose:
        print(f"[index] Wrote {len(all_rows)} total rows to {out_path} ({len(new_rows)} new, {len(existing_rows)} existing)")

if __name__ == '__main__':
    main()

