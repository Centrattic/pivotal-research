#!/usr/bin/env python3
"""
Test script for SAE probe implementation.
This script tests the basic functionality of the SAE probe without requiring
a full model or dataset setup.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

def test_sae_probe_import():
    """Test that SAE probe can be imported."""
    try:
        from src.probes.sae_probe import SAEProbe, SAEProbeNet
        print("✓ SAE probe imports successfully")
        return True
    except ImportError as e:
        print(f"✗ SAE probe import failed: {e}")
        return False

def test_sae_probe_net():
    """Test the SAEProbeNet neural network."""
    try:
        from src.probes.sae_probe import SAEProbeNet
        
        # Test different aggregation methods
        for aggregation in ["mean", "max", "last", "softmax"]:
            net = SAEProbeNet(sae_feature_dim=128, aggregation=aggregation, device="cpu")
            
            # Create dummy input
            batch_size, seq_len, feature_dim = 4, 10, 128
            x = torch.randn(batch_size, seq_len, feature_dim)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            
            # Test forward pass
            output = net(x, mask)
            assert output.shape == (batch_size,), f"Output shape mismatch for {aggregation}"
            
        print("✓ SAEProbeNet works with all aggregation methods")
        return True
    except Exception as e:
        print(f"✗ SAEProbeNet test failed: {e}")
        return False

def test_sae_probe_initialization():
    """Test SAE probe initialization."""
    try:
        from src.probes.sae_probe import SAEProbe
        
        # Test initialization without SAE ID (should fail)
        try:
            probe = SAEProbe(d_model=4096, device="cpu")
            print("✗ SAE probe should fail initialization without sae_id")
            return False
        except ValueError as e:
            if "sae_id must be provided" in str(e):
                print("✓ SAE probe correctly requires sae_id parameter")
                return True
            else:
                print(f"✗ Unexpected error during initialization: {e}")
                return False
        except Exception as e:
            print(f"✗ Unexpected error during initialization: {e}")
            return False
            
    except Exception as e:
        print(f"✗ SAE probe initialization test failed: {e}")
        return False

def test_config_integration():
    """Test that SAE probe configs are properly integrated."""
    try:
        from configs.probes import PROBE_CONFIGS
        
        # Check that SAE configs exist
        sae_configs = [k for k in PROBE_CONFIGS.keys() if k.startswith("sae")]
        expected_configs = [
            "sae_16k_l0_189_mean", "sae_16k_l0_189_max", "sae_16k_l0_189_last", "sae_16k_l0_189_softmax",
            "sae_131k_l0_153_mean", "sae_131k_l0_153_max", "sae_131k_l0_153_last", "sae_131k_l0_153_softmax",
            "sae_mean", "sae_max", "sae_last", "sae_softmax", "default_sae"
        ]
        
        for config in expected_configs:
            if config not in PROBE_CONFIGS:
                print(f"✗ Missing SAE config: {config}")
                return False
        
        # Check that specific SAE IDs are set in configs
        specific_configs = ["sae_16k_l0_189_mean", "sae_131k_l0_153_mean"]
        for config_name in specific_configs:
            config = PROBE_CONFIGS[config_name]
            if not hasattr(config, 'sae_id') or config.sae_id is None:
                print(f"✗ SAE config {config_name} missing sae_id")
                return False
        
        print(f"✓ Found {len(sae_configs)} SAE probe configurations with specific SAE IDs")
        return True
    except Exception as e:
        print(f"✗ Config integration test failed: {e}")
        return False

def test_utils_integration():
    """Test that SAE probe is integrated into utils."""
    try:
        from src.utils import get_probe_architecture
        
        # Test that SAE architecture is recognized
        config = {
            'aggregation': 'mean',
            'model_name': 'gemma-2-9b',
            'layer': 20,
            'sae_id': 'layer_20/width_16k/average_l0_189',
            'top_k_features': 128
        }
        
        # This should not raise an error for SAE architecture
        try:
            get_probe_architecture("sae_16k_l0_189_mean", d_model=4096, device="cpu", config=config)
            print("✓ SAE probe architecture is recognized in utils")
            return True
        except Exception as e:
            print(f"✗ Unexpected error in utils integration: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Utils integration test failed: {e}")
        return False

def test_sae_id_validation():
    """Test that SAE probe validates SAE ID properly."""
    try:
        from src.probes.sae_probe import SAEProbe
        
        # Test with valid SAE ID
        try:
            probe = SAEProbe(
                d_model=4096, 
                device="cpu", 
                sae_id="layer_20/width_16k/average_l0_189"
            )
            print("✓ SAE probe accepts valid SAE ID")
            return True
        except Exception as e:
            print(f"✗ Unexpected error with valid SAE ID: {e}")
            return False
            
    except Exception as e:
        print(f"✗ SAE ID validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing SAE Probe Implementation")
    print("=" * 40)
    
    tests = [
        test_sae_probe_import,
        test_sae_probe_net,
        test_sae_probe_initialization,
        test_config_integration,
        test_utils_integration,
        test_sae_id_validation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! SAE probe implementation is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 