#!/usr/bin/env python3
"""
Test script to verify multi-seed functionality.
"""

import sys
import os
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import get_effective_seeds
from utils import rebuild_suffix, resample_params_to_str

def test_seed_extraction():
    """Test seed extraction from different config formats."""
    print("Testing seed extraction...")
    
    # Test single seed (backward compatibility)
    config1 = {'seed': 42}
    seeds1 = get_effective_seeds(config1)
    print(f"Single seed config: {seeds1}")
    assert seeds1 == [42], f"Expected [42], got {seeds1}"
    
    # Test multiple seeds
    config2 = {'seeds': [42, 43, 44, 45, 46]}
    seeds2 = get_effective_seeds(config2)
    print(f"Multiple seeds config: {seeds2}")
    assert seeds2 == [42, 43, 44, 45, 46], f"Expected [42, 43, 44, 45, 46], got {seeds2}"
    
    # Test single value in seeds
    config3 = {'seeds': 42}
    seeds3 = get_effective_seeds(config3)
    print(f"Single value in seeds: {seeds3}")
    assert seeds3 == [42], f"Expected [42], got {seeds3}"
    
    # Test no seed specified
    config4 = {}
    seeds4 = get_effective_seeds(config4)
    print(f"No seed specified: {seeds4}")
    assert seeds4 == [42], f"Expected [42], got {seeds4}"
    
    print("✅ Seed extraction tests passed!")

def test_utils_functions():
    """Test utility functions with simplified seed handling."""
    print("\nTesting utility functions...")
    
    rebuild_config = {
        'class_counts': {0: 3750, 1: 250}
        # No seed field - only global seed is used
    }
    
    # Test rebuild_suffix
    suffix = rebuild_suffix(rebuild_config)
    print(f"rebuild_suffix: {suffix}")
    assert "class0_3750_class1_250" in suffix, f"Expected 'class0_3750_class1_250' in suffix, got {suffix}"
    assert "seed" not in suffix, f"Expected no 'seed' in suffix, got {suffix}"
    
    # Test resample_params_to_str
    result_str = resample_params_to_str(rebuild_config)
    print(f"resample_params_to_str: {result_str}")
    assert "class0_3750_class1_250" in result_str, f"Expected 'class0_3750_class1_250' in string, got {result_str}"
    assert "seed" not in result_str, f"Expected no 'seed' in string, got {result_str}"
    
    # Test with class_percents
    rebuild_config_percents = {
        'class_percents': {0: 0.8, 1: 0.2},
        'total_samples': 1000
    }
    
    suffix_percents = rebuild_suffix(rebuild_config_percents)
    print(f"rebuild_suffix with percents: {suffix_percents}")
    assert "class0_80pct_class1_20pct_total1000" in suffix_percents, f"Expected 'class0_80pct_class1_20pct_total1000' in suffix, got {suffix_percents}"
    
    # Test with None
    suffix_none = rebuild_suffix(None)
    print(f"rebuild_suffix with None: {suffix_none}")
    assert suffix_none == "original", f"Expected 'original', got {suffix_none}"
    
    str_none = resample_params_to_str(None)
    print(f"resample_params_to_str with None: {str_none}")
    assert str_none == "original", f"Expected 'original', got {str_none}"
    
    print("✅ Utility function tests passed!")

def test_config_file():
    """Test loading the actual config file."""
    print("\nTesting config file loading...")
    
    config_path = Path("configs/spam_exp_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        seeds = get_effective_seeds(config)
        print(f"Config file seeds: {seeds}")
        assert len(seeds) == 5, f"Expected 5 seeds, got {len(seeds)}"
        assert seeds == [42, 43, 44, 45, 46], f"Expected [42, 43, 44, 45, 46], got {seeds}"
        
        # Check that rebuild_config entries don't have seeds
        for experiment in config.get('experiments', []):
            if 'rebuild_config' in experiment:
                for group_name, group_configs in experiment['rebuild_config'].items():
                    for config_entry in group_configs:
                        assert 'seed' not in config_entry, f"Found 'seed' in rebuild_config entry: {config_entry}"
        
        print("✅ Config file test passed!")
    else:
        print("⚠️  Config file not found, skipping config file test")

if __name__ == "__main__":
    print("🧪 Running multi-seed functionality tests...\n")
    
    test_seed_extraction()
    test_utils_functions()
    test_config_file()
    
    print("\n🎉 All tests passed! Multi-seed functionality is working correctly.") 