#!/usr/bin/env python3
import json
import sys
import argparse
from sklearn.metrics import roc_auc_score

def compute_auc(path: str) -> float:
    with open(path, "r") as f:
        data = json.load(f)
    scores = data["scores"]["scores"]
    labels = data["scores"]["labels"]
    if len(set(labels)) < 2:
        raise ValueError("AUC is undefined: labels contain only one class")
    return roc_auc_score(labels, scores)

def main():
    parser = argparse.ArgumentParser(description="Compute ROC AUC from results JSON")
    parser.add_argument("path", help="Path to results JSON")
    args = parser.parse_args()
    try:
        auc = compute_auc(args.path)
        print(f"{auc:.6f}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
