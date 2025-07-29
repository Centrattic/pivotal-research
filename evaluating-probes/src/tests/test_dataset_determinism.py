#!/usr/bin/env python3
"""
Test script to verify that dataset creation is deterministic for a given seed.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import Dataset

def test_dataset_determinism():
    """Test that dataset creation is deterministic for a given seed."""
    print("Testing dataset determinism...")
    
    # Test parameters
    dataset_name = "4_hist_fig_ismale"  # Use a simple dataset
    seed = 42
    
    # Create dataset twice with the same seed
    print(f"Creating dataset '{dataset_name}' with seed {seed}...")
    
    # First creation
    original_ds1 = Dataset(dataset_name, model=None, device="cpu", seed=seed)
    ds1 = Dataset.build_imbalanced_train_balanced_eval(original_ds1, val_size=0.10, test_size=0.15, seed=seed)
    X_test1, y_test1 = ds1.get_test_set()
    
    # Second creation
    original_ds2 = Dataset(dataset_name, model=None, device="cpu", seed=seed)
    ds2 = Dataset.build_imbalanced_train_balanced_eval(original_ds2, val_size=0.10, test_size=0.15, seed=seed)
    X_test2, y_test2 = ds2.get_test_set()
    
    # Check if they're identical
    print(f"Test set 1 size: {len(X_test1)}")
    print(f"Test set 2 size: {len(X_test2)}")
    print(f"Test set 1 class distribution: {dict(zip(*np.unique(y_test1, return_counts=True)))}")
    print(f"Test set 2 class distribution: {dict(zip(*np.unique(y_test2, return_counts=True)))}")
    
    # Check if the test sets are identical
    if len(X_test1) == len(X_test2) and len(y_test1) == len(y_test2):
        if np.array_equal(y_test1, y_test2):
            print("✅ SUCCESS: Test sets are identical!")
            print("✅ Dataset creation is deterministic for the given seed.")
            return True
        else:
            print("❌ FAILURE: Test set labels are different!")
            print(f"Labels 1: {y_test1[:10]}...")
            print(f"Labels 2: {y_test2[:10]}...")
            return False
    else:
        print("❌ FAILURE: Test set sizes are different!")
        return False

def test_static_method_determinism():
    """Test that the static method build_imbalanced_train_balanced_eval is deterministic."""
    print("\nTesting static method determinism...")
    
    # Test parameters
    dataset_name = "4_hist_fig_ismale"
    seed = 42
    
    # Create original dataset
    original_ds = Dataset(dataset_name, model=None, device="cpu", seed=seed)
    
    # First creation using static method
    ds1 = Dataset.build_imbalanced_train_balanced_eval(
        original_ds, 
        val_size=0.10, 
        test_size=0.15, 
        seed=seed
    )
    X_test1, y_test1 = ds1.get_test_set()
    
    # Second creation using static method
    ds2 = Dataset.build_imbalanced_train_balanced_eval(
        original_ds, 
        val_size=0.10, 
        test_size=0.15, 
        seed=seed
    )
    X_test2, y_test2 = ds2.get_test_set()
    
    # Check if they're identical
    print(f"Static method - Test set 1 size: {len(X_test1)}")
    print(f"Static method - Test set 2 size: {len(X_test2)}")
    print(f"Static method - Test set 1 class distribution: {dict(zip(*np.unique(y_test1, return_counts=True)))}")
    print(f"Static method - Test set 2 class distribution: {dict(zip(*np.unique(y_test2, return_counts=True)))}")
    
    # Check if the test sets are identical
    if len(X_test1) == len(X_test2) and len(y_test1) == len(y_test2):
        if np.array_equal(y_test1, y_test2):
            print("✅ SUCCESS: Static method test sets are identical!")
            print("✅ Static method is deterministic for the given seed.")
            return True
        else:
            print("❌ FAILURE: Static method test set labels are different!")
            print(f"Labels 1: {y_test1[:10]}...")
            print(f"Labels 2: {y_test2[:10]}...")
            return False
    else:
        print("❌ FAILURE: Static method test set sizes are different!")
        return False

if __name__ == "__main__":
    success1 = test_dataset_determinism()
    success2 = test_static_method_determinism()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1) 