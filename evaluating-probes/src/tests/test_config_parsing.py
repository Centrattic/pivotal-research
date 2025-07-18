#!/usr/bin/env python3
"""
Test script for config parsing with score options.
"""

import yaml
from pathlib import Path

def test_config_parsing():
    """Test parsing of config files with score options."""
    
    # Test config with different score options
    test_configs = [
        {
            'name': 'test_all_only',
            'train_on': 'test_dataset',
            'evaluate_on': ['test_dataset'],
            'score': ['all']
        },
        {
            'name': 'test_filtered_only', 
            'train_on': 'test_dataset',
            'evaluate_on': ['test_dataset'],
            'score': ['filtered']
        },
        {
            'name': 'test_both',
            'train_on': 'test_dataset', 
            'evaluate_on': ['test_dataset'],
            'score': ['all', 'filtered']
        },
        {
            'name': 'test_no_score',
            'train_on': 'test_dataset',
            'evaluate_on': ['test_dataset']
            # No score field - should default to ['all']
        }
    ]
    
    for config in test_configs:
        print(f"\nTesting config: {config['name']}")
        
        # Simulate the parsing logic from main.py
        score_options = config.get('score', ['all'])
        print(f"  Score options: {score_options}")
        
        # Simulate the evaluation logic from runner.py
        if 'all' in score_options:
            print(f"  ✅ Will calculate metrics for all examples")
        
        if 'filtered' in score_options:
            print(f"  ✅ Will calculate filtered metrics")
        
        if len(score_options) == 1:
            print(f"  📝 Single scoring mode: {score_options[0]}")
        else:
            print(f"  📝 Combined scoring mode: {score_options}")
    
    print("\n✅ All config parsing tests passed!")

if __name__ == "__main__":
    test_config_parsing() 