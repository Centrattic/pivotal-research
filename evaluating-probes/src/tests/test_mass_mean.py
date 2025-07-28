#!/usr/bin/env python3
"""
Simple test script for the Mass Mean probe implementation.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.probes import MassMeanProbe

def test_mass_mean_probe():
    """Test the mass-mean probe implementation."""
    print("Testing Mass Mean Probe...")
    
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic activations and labels
    n_samples = 1000
    seq_len = 10
    d_model = 64
    
    # Create two classes with different means
    X = np.random.randn(n_samples, seq_len, d_model)
    y = np.random.randint(0, 2, n_samples)
    
    # Add class-specific patterns
    class_0_mean = np.random.randn(d_model) * 0.1
    class_1_mean = np.random.randn(d_model) * 0.1 + 0.5  # Offset to make classes separable
    
    for i in range(n_samples):
        if y[i] == 0:
            X[i] += class_0_mean
        else:
            X[i] += class_1_mean
    
    # Create mask (all tokens are valid)
    mask = np.ones((n_samples, seq_len), dtype=bool)
    
    # Test basic mass-mean probe
    print("\n1. Testing basic mass-mean probe...")
    probe_basic = MassMeanProbe(d_model=d_model, device="cpu", use_iid=False)
    probe_basic.fit(X, y, mask=mask)
    
    # Test predictions
    predictions = probe_basic.predict(X[:10], mask=mask[:10])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Test IID mass-mean probe
    print("\n2. Testing IID mass-mean probe...")
    probe_iid = MassMeanProbe(d_model=d_model, device="cpu", use_iid=True)
    probe_iid.fit(X, y, mask=mask)
    
    # Test predictions
    predictions_iid = probe_iid.predict(X[:10], mask=mask[:10])
    print(f"IID Predictions shape: {predictions_iid.shape}")
    print(f"IID Predictions: {predictions_iid}")
    
    # Test scoring
    print("\n3. Testing scoring...")
    scores_basic = probe_basic.score(X, y, mask=mask)
    scores_iid = probe_iid.score(X, y, mask=mask)
    
    print(f"Basic mass-mean scores: {scores_basic}")
    print(f"IID mass-mean scores: {scores_iid}")
    
    # Test save/load
    print("\n4. Testing save/load...")
    save_path = Path("test_mass_mean_probe.pt")
    probe_basic.save_state(save_path)
    
    # Load the probe
    probe_loaded = MassMeanProbe(d_model=d_model, device="cpu", use_iid=False)
    probe_loaded.load_state(save_path)
    
    # Test that predictions are the same
    pred_original = probe_basic.predict(X[:5], mask=mask[:5])
    pred_loaded = probe_loaded.predict(X[:5], mask=mask[:5])
    
    print(f"Original predictions: {pred_original}")
    print(f"Loaded predictions: {pred_loaded}")
    print(f"Predictions match: {np.allclose(pred_original, pred_loaded)}")
    
    # Clean up
    save_path.unlink(missing_ok=True)
    
    print("\n✅ Mass Mean Probe test completed successfully!")

if __name__ == "__main__":
    test_mass_mean_probe() 