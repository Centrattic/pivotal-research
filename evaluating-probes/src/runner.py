# src/runner.py
import json
from pathlib import Path
from dataclasses import asdict

from src.data import DataLoader
from src.activations import ActivationManager
from src.probes import LinearProbe, AttentionProbe
from configs.probes import PROBE_CONFIGS

def get_probe_architecture(architecture_name: str, d_model: int):
    """Factory function to create a probe instance."""
    if architecture_name == "linear":
        return LinearProbe(d_model=d_model)
    if architecture_name == "attention":
        return AttentionProbe(d_model=d_model)
    raise ValueError(f"Unknown architecture: {architecture_name}")

def run_single_experiment(
    model_name: str,
    d_model: int,
    dataset_name: str,
    layer: int,
    component: str,
    architecture_config: dict,
    aggregation: str,
    device: str,
    use_cache: bool,
    results_dir: Path,
    cache_dir: Path,
):
    """Orchestrates a single probing experiment using the new methodology."""
    architecture_name = architecture_config['name']
    config_name = architecture_config['config_name']

    # The 'attention' architecture has its own implicit aggregation.
    # We arbitrarily pick one aggregation from the list to run it once and skip the others.
    if architecture_name == "attention" and aggregation != "mean":
        print(f"  - ⏭️  Skipping redundant aggregation '{aggregation}' for attention architecture.")
        return
    
    # The 'attention' aggregation method is only for the attention architecture.
    if architecture_name == "linear" and aggregation == "attention":
        print(f"  - ⏭️  Skipping 'attention' aggregation for linear architecture.")
        return

    # 1. Load Data
    data_loader = DataLoader(dataset_name)
    X_train_text, y_train, X_test_text, y_test = data_loader.split_text_and_labels()

    # 2. Get Activations: compute if don't exist, else load
    act_manager = ActivationManager(model_name, device, d_model=d_model)
    dataset_cache_dir = cache_dir / dataset_name
    
    train_acts = act_manager.get_activations(X_train_text, layer, component, use_cache, dataset_cache_dir)
    test_acts = act_manager.get_activations(X_test_text, layer, component, use_cache, dataset_cache_dir)

    # 3. Instantiate Probe and get its config
    probe = get_probe_architecture(architecture_name, d_model=d_model)
    fit_params = asdict(PROBE_CONFIGS[config_name])

    # 4. Train Probe
    probe.fit(train_acts, y_train, aggregation=aggregation, **fit_params)

    # 5. Evaluate and Save Results
    metrics = probe.score(test_acts, y_test, aggregation=aggregation)
    
    # Define a unique name for the results
    agg_name = "attention" if architecture_name == "attention" else aggregation
    filename_prefix = f"L{layer}_{component}_{architecture_name}_{agg_name}"
    dataset_results_dir = results_dir / dataset_name
    dataset_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metadata = {
        "metrics": metrics,
        "layer": layer,
        "component": component,
        "architecture": architecture_name,
        "aggregation": agg_name,
        "config": fit_params
    }
    with open(dataset_results_dir / f"{filename_prefix}_results.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save probe state
    probe.save_state(dataset_results_dir / f"{filename_prefix}_state.npz")
    
    print(f"  - ✅ Success! Test Accuracy: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}")