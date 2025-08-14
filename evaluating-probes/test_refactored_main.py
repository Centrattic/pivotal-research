#!/usr/bin/env python3
"""
Test script for the refactored main.py
This script tests the basic functionality without running the full pipeline.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    try:
        from main import (
            extract_activations_for_dataset,
            train_single_probe,
            evaluate_probe_on_dataset,
            process_probe_job,
            run_llm_upsampling_from_config
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test that config loading works correctly."""
    try:
        import yaml
        config_path = Path("configs/spam_exp_fast_config.yaml")
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['run_name', 'model_name', 'seeds', 'layers', 'components', 'architectures', 'experiments']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ Config loading successful")
        print(f"   - Run name: {config['run_name']}")
        print(f"   - Model: {config['model_name']}")
        print(f"   - Seeds: {config['seeds']}")
        print(f"   - On policy: {config.get('on_policy', False)}")
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_llm_upsampling_wrapper():
    """Test the LLM upsampling wrapper function."""
    try:
        from main import run_llm_upsampling_from_config
        import yaml
        
        # Load a test config
        config_path = Path("configs/spam_exp_fast_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Test without LLM upsampling
        run_llm_upsampling_from_config(config)
        
        # Test with LLM upsampling
        config['llm_upsampling'] = True
        run_llm_upsampling_from_config(config)
        
        print("✅ LLM upsampling wrapper test successful")
        return True
    except Exception as e:
        print(f"❌ LLM upsampling wrapper error: {e}")
        return False

def test_utils_functions():
    """Test that the updated utils functions work correctly."""
    try:
        from utils import get_effective_seeds, resample_params_to_str
        
        # Test get_effective_seeds
        config = {'seeds': [42, 43, 44]}
        seeds = get_effective_seeds(config)
        assert seeds == [42, 43, 44], f"Expected [42, 43, 44], got {seeds}"
        
        # Test resample_params_to_str
        params = {'class_counts': {0: 100, 1: 50}}
        result = resample_params_to_str(params)
        assert 'class0_100' in result, f"Expected 'class0_100' in result, got {result}"
        
        print("✅ Utils functions test successful")
        return True
    except Exception as e:
        print(f"❌ Utils functions error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing refactored main.py...")
    print("=" * 50)
    
    tests = [
        ("Import test", test_imports),
        ("Config loading test", test_config_loading),
        ("LLM upsampling wrapper test", test_llm_upsampling_wrapper),
        ("Utils functions test", test_utils_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored main.py should work correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
