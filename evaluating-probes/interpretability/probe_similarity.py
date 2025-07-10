# interpretability/probe_similarity.py
import torch
import json
from pathlib import Path
from itertools import combinations
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    # Example: Compare all probes trained on the 'hist_fig_ismale' dataset
    # for the 'pythia_70m_initial_run'
    
    run_name = "pythia_70m_initial_run"
    dataset_to_compare = "hist_fig_ismale"
    
    results_dir = Path("results") / run_name / dataset_to_compare
    if not results_dir.exists():
        print(f"No results found for {dataset_to_compare} in run {run_name}")
        return

    # Load all probe states (.pt files)
    probe_states = {}
    for probe_file in results_dir.glob("*.pt"):
        probe_name = probe_file.stem
        state = torch.load(probe_file)
        # Assuming the state dict contains a 'theta' vector
        if 'theta' in state:
            probe_states[probe_name] = state['theta'].numpy()

    print(f"Found {len(probe_states)} probes to compare for dataset '{dataset_to_compare}'.")
    
    # Calculate pairwise cosine similarity
    for (name1, theta1), (name2, theta2) in combinations(probe_states.items(), 2):
        sim = cosine_similarity(theta1, theta2)
        print(f"- Sim({name1}, {name2}): {sim:.3f}")

if __name__ == "__main__":
    main()