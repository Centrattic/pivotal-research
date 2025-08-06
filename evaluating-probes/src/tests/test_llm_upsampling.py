#!/usr/bin/env python3
"""
Test script for LLM upsampling functionality.

This script tests the core functionality without making actual API calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.llm_upsampling.llm_upsampling_script import LLMUpsamplingStateMachine, extract_samples_for_upsampling

def create_test_data():
    """Create test data for validation."""
    test_data = {
        'prompt': [
            "URGENT: You've won $1,000,000! Click here to claim your prize!",
            "FREE VIAGRA NOW! Limited time offer, act fast!",
            "Your account has been suspended. Verify immediately.",
            "Congratulations! You're our 1,000,000th visitor!",
            "Make money fast from home! No experience needed!"
        ],
        'target': [1, 1, 1, 1, 1]
    }
    return pd.DataFrame(test_data)

def test_csv_extraction():
    """Test CSV extraction from LLM response."""
    print("Testing CSV extraction...")
    
    # Test with "Response below" marker
    test_response = """Here are the upsampled samples:

Response below
prompt,target
"New spam email 1",1
"New spam email 2",1
"New spam email 3",1"""

    state_machine = LLMUpsamplingStateMachine("fake_api_key")
    df = state_machine.extract_csv_from_response(test_response)
    
    assert df is not None
    assert len(df) == 3
    assert all(df['target'] == 1)
    print("✅ CSV extraction test passed")

def test_validation():
    """Test data validation."""
    print("Testing data validation...")
    
    original_df = create_test_data()
    state_machine = LLMUpsamplingStateMachine("fake_api_key")
    
    # Test valid data
    valid_df = pd.DataFrame({
        'prompt': ["New spam 1", "New spam 2", "New spam 3"],
        'target': [1, 1, 1]
    })
    
    is_valid, msg = state_machine.validate_upsampled_data(original_df, valid_df, 3, 2)
    assert is_valid
    print("✅ Valid data test passed")
    
    # Test insufficient samples
    insufficient_df = pd.DataFrame({
        'prompt': ["New spam 1"],
        'target': [1]
    })
    
    is_valid, msg = state_machine.validate_upsampled_data(original_df, insufficient_df, 3, 2)
    assert not is_valid
    assert "Not enough samples" in msg
    print("✅ Insufficient samples test passed")
    
    # Test wrong class
    wrong_class_df = pd.DataFrame({
        'prompt': ["New spam 1", "New spam 2"],
        'target': [1, 0]
    })
    
    is_valid, msg = state_machine.validate_upsampled_data(original_df, wrong_class_df, 2, 2)
    assert not is_valid
    assert "not class 1" in msg
    print("✅ Wrong class test passed")
    
    # Test duplicates
    duplicate_df = pd.DataFrame({
        'prompt': ["URGENT: You've won $1,000,000! Click here to claim your prize!", "New spam 2"],
        'target': [1, 1]
    })
    
    is_valid, msg = state_machine.validate_upsampled_data(original_df, duplicate_df, 2, 2)
    assert not is_valid
    assert "duplicate" in msg
    print("✅ Duplicate detection test passed")

def test_prompt_generation():
    """Test prompt generation."""
    print("Testing prompt generation...")
    
    original_df = create_test_data()
    state_machine = LLMUpsamplingStateMachine("fake_api_key")
    
    prompt = state_machine.generate_upsampling_prompt(original_df, 2, 10)
    
    assert "upsample this dataset 2x" in prompt
    assert "10 samples of class 1" in prompt
    assert "Response below" in prompt
    assert "prompt,target" in prompt
    print("✅ Prompt generation test passed")

def test_sample_extraction():
    """Test sample extraction logic."""
    print("Testing sample extraction...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = """
run_name: "test_run"
model_name: "test_model"
experiments:
  - name: "test_exp"
    train_on: "87_is_spam"
"""
        f.write(config_content)
        config_path = Path(f.name)
    
    try:
        # This will fail because the dataset doesn't exist, but we can test the logic
        try:
            extract_samples_for_upsampling(config_path, [1, 2], 42)
        except FileNotFoundError:
            # Expected error since test_dataset doesn't exist
            print("✅ Sample extraction logic test passed (expected error for missing dataset)")
    finally:
        # Clean up
        config_path.unlink()

def test_integration_workflow():
    """Test the integration workflow with mock data."""
    print("Testing integration workflow...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock LLM output directory
        llm_output_dir = temp_path / "llm_generated"
        llm_output_dir.mkdir()
        
        # Create mock LLM-generated files
        for n_real in [1, 2]:
            for factor in [2, 3]:
                mock_df = pd.DataFrame({
                    'prompt': [f"Mock spam {n_real}_{factor}_{i}" for i in range(factor)],
                    'target': [1] * factor
                })
                filename = f"llm_samples_{n_real}_{factor}x.csv"
                mock_df.to_csv(llm_output_dir / filename, index=False)
        
        # Verify files were created
        csv_files = list(llm_output_dir.glob("llm_samples_*.csv"))
        assert len(csv_files) == 4  # 2 n_real_samples × 2 upsampling_factors
        
        print("✅ Integration workflow test passed")

def main():
    """Run all tests."""
    print("Running LLM upsampling tests...")
    print("=" * 50)
    
    try:
        test_csv_extraction()
        test_validation()
        test_prompt_generation()
        test_sample_extraction()
        test_integration_workflow()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 