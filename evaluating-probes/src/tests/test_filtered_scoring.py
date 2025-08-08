#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append('src')

from src.probes import LinearProbe
from src.data import Dataset
from transformer_lens import HookedTransformer

def test_filtered_scoring():
    print("Testing filtered scoring fix...")
    
    # Create a simple test setup
    device = "cpu"
    d_model = 64
    n_samples = 100
    seq_len = 10
    
    # Create synthetic data
    X = np.random.randn(n_samples, seq_len, d_model).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    mask = np.ones((n_samples, seq_len), dtype=bool)
    
    # Create a probe
    probe = LinearProbe(d_model=d_model, device=device, task_type="classification", aggregation="mean")
    
    # Train the probe briefly
    print("Training probe...")
    probe.fit(X, y, mask=mask, epochs=5, lr=1e-2, batch_size=32, verbose=False)
    
    # Test regular scoring
    print("\nTesting regular scoring...")
    regular_metrics = probe.score(X, y, mask)
    print(f"Regular metrics: {regular_metrics}")
    
    # Test filtered scoring with different thresholds
    print("\nTesting filtered scoring...")
    thresholds = [0.1, 0.5, 1.0, 2.0]
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        filtered_metrics = probe.score_filtered(
            X, y, "test_dataset", Path("results"), 
            seed=42, threshold_class_0=threshold, threshold_class_1=threshold, test_size=0.15, mask=mask
        )
        print(f"Filtered metrics: {filtered_metrics}")
        
        # Check that filtered metrics are different from regular metrics
        if filtered_metrics.get("filtered", False):
            print("😄 Filtered scoring worked!")
            print(f"   Original size: {filtered_metrics['original_size']}")
            print(f"   Filtered size: {filtered_metrics['filtered_size']}")
            print(f"   Removed: {filtered_metrics['removed_count']}")
            print(f"   Filtering %: {filtered_metrics['filtering_percentage']:.1f}%")
        else:
            print("😢 Filtered scoring failed")

if __name__ == "__main__":
    test_filtered_scoring() 