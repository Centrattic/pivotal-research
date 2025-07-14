# interpretability/generate_viz_data.py
import json
from pathlib import Path
import numpy as np
from itertools import combinations

def get_probe_vector(probe_state: dict) -> np.ndarray | None:
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
    Scans the results directory, aggregates performance data and probe vectors,
    and saves it into a single JSON file for visualization.
    """
    print("Starting data generation for visualizations...")
    
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
            probe_name = state_file.stem.replace("_state", "")
            train_dataset = "_".join(state_file.parent.name.split("_")[1:])

            vector = get_probe_vector(data)
            if vector is not None:
                if train_dataset not in all_data["probe_vectors"]:
                    all_data["probe_vectors"][train_dataset] = {}
                all_data["probe_vectors"][train_dataset][probe_name] = vector.tolist()
        except Exception as e:
            print(f"Could not process state file {state_file}: {e}")

    # --- 3. Pre-calculate cosine similarities ---
    print("Calculating cosine similarities...")
    all_data["similarities"] = {}
    for dataset, probes in all_data["probe_vectors"].items():
        all_data["similarities"][dataset] = []
        probe_names = list(probes.keys())

        # Calculate mean vector for the probes on this dataset
        if probes:
            mean_vec = np.mean(list(probes.values()), axis=0)
        else:
            continue

        # Get single_all vector if it exists
        single_all_probes = all_data["probe_vectors"].get("single_all", {})
        single_all_probe_name = next((p for p in single_all_probes if "linear" in p), None)
        single_all_vec = np.array(single_all_probes[single_all_probe_name]) if single_all_probe_name else None

        for p_name, p_vec_list in probes.items():
            p_vec = np.array(p_vec_list)
            # Sim vs mean
            sim_mean = np.dot(p_vec, mean_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(mean_vec))
            all_data["similarities"][dataset].append({"probe1": p_name, "probe2": "MEAN_PROBE", "similarity": float(sim_mean)})
            # Sim vs single_all
            if single_all_vec is not None:
                sim_all = np.dot(p_vec, single_all_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(single_all_vec))
                all_data["similarities"][dataset].append({"probe1": p_name, "probe2": "SINGLE_ALL_PROBE", "similarity": float(sim_all)})

        # Pairwise similarities
        for p1_name, p2_name in combinations(probe_names, 2):
            v1, v2 = np.array(probes[p1_name]), np.array(probes[p2_name])
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            all_data["similarities"][dataset].append({"probe1": p1_name, "probe2": p2_name, "similarity": float(sim)})

    # --- 4. Save the aggregated data ---
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Visualization data successfully saved to {output_path}")

if __name__ == "__main__":
    main()
