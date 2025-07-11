# interpretability/generate_viz_data.py
import json
from pathlib import Path
import numpy as np
from itertools import combinations

def get_probe_vector(probe_state: dict) -> np.ndarray:
    """Extracts the primary direction vector from a probe state, handling different architectures."""
    if 'theta' in probe_state:
        # For LinearProbe, theta can be (D,) or (K, D). We take the norm for multiclass.
        theta = probe_state['theta']
        if theta.ndim > 1:
            return np.linalg.norm(theta, axis=0)
        return theta
    elif 'theta_q' in probe_state:
        # For AttentionProbe, theta_q is the most representative direction.
        return probe_state['theta_q']
    return None

def main():
    """
    Scans the results directory, aggregates all experiment data and probe vectors,
    and saves it into a single JSON file for visualization.
    """
    print("Starting data generation for visualizations...")
    
    # Assumes this script is in 'interpretability' and 'results' is in the parent dir
    results_root = Path(__file__).parent.parent / "results"
    output_path = Path(__file__).parent / "visualization_data.json"
    
    all_data = {
        "performance": [],
        "probe_vectors": {} # { "dataset_name": { "probe_name": [vector] } }
    }

    if not results_root.exists():
        print(f"Error: Results directory not found at {results_root}")
        return

    # --- 1. Gather all performance results ---
    for result_file in results_root.glob("**/*_results.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_data["performance"].append(data)

    # --- 2. Gather all probe vectors ---
    for state_file in results_root.glob("**/*_state.npz"):
        try:
            data = np.load(state_file)
            # The probe name is the state file's name without the suffix
            probe_name = state_file.stem.replace("_state", "")
            # The dataset is part of the parent directory name, e.g., "train_64_..."
            train_dataset = [p for p in state_file.parent.name.split("_") if p.isdigit()]
            if not train_dataset: continue
            
            # Reconstruct the dataset name from the directory
            dataset_name = "_".join(state_file.parent.name.split("_")[1:])

            vector = get_probe_vector(data)
            if vector is not None:
                if dataset_name not in all_data["probe_vectors"]:
                    all_data["probe_vectors"][dataset_name] = {}
                all_data["probe_vectors"][dataset_name][probe_name] = vector.tolist()
        except Exception as e:
            print(f"Could not process state file {state_file}: {e}")

    # --- 3. Pre-calculate cosine similarities ---
    all_data["similarities"] = {}
    for dataset, probes in all_data["probe_vectors"].items():
        all_data["similarities"][dataset] = []
        probe_names = list(probes.keys())
        for p1_name, p2_name in combinations(probe_names, 2):
            v1 = np.array(probes[p1_name])
            v2 = np.array(probes[p2_name])
            
            # Cosine similarity calculation
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            all_data["similarities"][dataset].append({
                "probe1": p1_name,
                "probe2": p2_name,
                "similarity": float(sim)
            })

    # --- 4. Save the aggregated data ---
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Visualization data successfully saved to {output_path}")

if __name__ == "__main__":
    main()
