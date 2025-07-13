# interpretability/generate_viz_data.py

import json
from pathlib import Path
import numpy as np
from itertools import combinations
import sys

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from src.data import Dataset
from src.probes import LinearProbe, AttentionProbe
from src.activations import ActivationManager
from src.logger import Logger

# --- Helper Functions ---
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

def get_probe_instance(probe_name: str, state_file_path: Path, d_model: int = 512):
    """Loads a probe's state and returns a probe object."""
    data = np.load(state_file_path)
    arch = "attention" if "attention" in probe_name else "linear"
    
    if arch == "linear":
        probe = LinearProbe(d_model)
    else:
        probe = AttentionProbe(d_model)
    
    probe.load_state(state_file_path, Logger(Path("temp_log.txt"))) # Use a temp logger
    return probe

def main():
    print("Starting data generation for visualizations...")
    
    results_root = Path(__file__).parent.parent / "results"
    output_path = Path(__file__).parent / "visualization_data.json"
    
    all_data = {
        "performance": [],
        "probe_vectors": {},
        "raw_scores": {}, # { eval_dataset: { probe_name: [scores] } }
    }

    # --- 1. Gather all performance results ---
    for result_file in results_root.glob("**/*_results.json"):
        with open(result_file, 'r') as f:
            all_data["performance"].append(json.load(f))

    # --- 2. Gather probe vectors and calculate raw scores ---
    # This is slow as it requires getting activations, but is the most robust method.
    print("Calculating raw scores for violin plots (this may take a while)...")
    act_managers = {} # Cache activation managers

    for state_file in results_root.glob("**/*_state.npz"):
        try:
            probe_name = state_file.stem.replace("_state", "")
            train_dataset = "_".join(state_file.parent.name.split("_")[1:])
            
            # Get probe vector
            vector = get_probe_vector(np.load(state_file))
            if vector is not None:
                if train_dataset not in all_data["probe_vectors"]:
                    all_data["probe_vectors"][train_dataset] = {}
                all_data["probe_vectors"][train_dataset][probe_name] = vector.tolist()

            # Get raw scores for every evaluation of this probe
            probe_instance = get_probe_instance(probe_name, state_file)
            for result_file in state_file.parent.glob(f"*__{probe_name}_results.json"):
                with open(result_file, 'r') as f:
                    meta = json.load(f)
                
                eval_dataset_name = meta['eval_dataset']
                if eval_dataset_name not in all_data['raw_scores']:
                    all_data['raw_scores'][eval_dataset_name] = {}

                # Get activations for this eval set
                eval_data = Dataset(eval_dataset_name, seed=meta['seed'])
                model_name = "EleutherAI/pythia-70m" # Assuming this for now
                if model_name not in act_managers:
                    act_managers[model_name] = {}
                if eval_data.max_len not in act_managers[model_name]:
                     act_managers[model_name][eval_data.max_len] = ActivationManager(model_name, 'cpu', 512, eval_data.max_len)
                
                act_manager = act_managers[model_name][eval_data.max_len]
                cache_dir = Path(__file__).parent.parent / "activation_cache" / model_name
                
                X_test, _ = eval_data.get_test_set()
                acts = act_manager.get_activations(X_test, meta['layer'], meta['component'], True, cache_dir / eval_dataset_name, Logger(Path("temp_log.txt")))
                
                # Predict and store raw scores (logits)
                scores = probe_instance.predict(acts, meta['aggregation'])
                all_data['raw_scores'][eval_dataset_name][probe_name] = scores.tolist()

        except Exception as e:
            print(f"Could not process state file {state_file}: {e}")

    # --- 3. Pre-calculate cosine similarities ---
    print("Calculating cosine similarities...")
    all_data["similarities"] = {}
    for dataset, probes in all_data["probe_vectors"].items():
        all_data["similarities"][dataset] = []
        probe_names = list(probes.keys())

        # Calculate mean vector
        mean_vec = np.mean(list(probes.values()), axis=0)
        
        # Get single_all vector if it exists
        single_all_probe_name = [p for p in all_data["probe_vectors"].get("single_all", {}) if "linear" in p] # Find a linear single_all probe
        single_all_vec = np.array(all_data["probe_vectors"]["single_all"][single_all_probe_name[0]]) if single_all_probe_name else None

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
        json.dump(all_data, f) # Use compact format for smaller file size

    print(f"✅ Visualization data successfully saved to {output_path}")

if __name__ == "__main__":
    main()
