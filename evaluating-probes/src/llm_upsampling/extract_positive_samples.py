#!/usr/bin/env python3
"""
Extract positive samples from datasets for LLM upsampling.

This script extracts positive samples from the original dataset based on the
n_real_pos values specified in the LLM upsampling configuration. These CSV files
will be used as input to the Gemini LLM for generating synthetic positive samples.

Usage:
    python -m src.llm_upsampling.extract_positive_samples -c spam_exp
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from src.data import Dataset
from transformer_lens import HookedTransformer


def extract_positive_samples_for_llm(config_path):
    """
    Extract positive samples from datasets for LLM upsampling.
    
    Args:
        config_path: Path to the config YAML file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    run_name = config['run_name']
    model_name = config['model_name']
    device = config.get('device', 'cpu')
    seed = config.get('seeds', [42])[0]  # Use first seed for extraction, should alwasy be constant
    
    # Create output directory
    output_dir = Path('results') / run_name / 'csvs_pass_to_llm'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting positive samples for LLM upsampling...")
    print(f"Output directory: {output_dir}")
    
    # Load model
    # model = HookedTransformer.from_pretrained(model_name, device=device)
    
    # Find LLM upsampling experiments
    for experiment in config['experiments']:
        rebuild_config = experiment.get('rebuild_config', {})
        
        if 'llm_upsampling_experiments' not in rebuild_config:
            continue
            
        train_on = experiment['train_on']
        print(f"\nProcessing experiment: {experiment['name']}")
        print(f"Training dataset: {train_on}")
        
        # Load original dataset
        print(f"Loading original dataset: {train_on}")
        orig_ds = Dataset(train_on, model=None, device=device, seed=seed)
        
        # Get unique n_real_pos values from the config
        llm_configs = rebuild_config['llm_upsampling_experiments']
        n_real_pos_values = set()
        
        for llm_config in llm_configs:
            if 'llm_upsampling' in llm_config and llm_config['llm_upsampling']:
                n_real_pos = llm_config.get('n_real_pos')
                if n_real_pos is not None:
                    n_real_pos_values.add(n_real_pos)
        
        n_real_pos_values = sorted(list(n_real_pos_values))
        print(f"Found n_real_pos values: {n_real_pos_values}")
        
        # Extract positive samples for each n_real_pos value
        for n_real_pos in n_real_pos_values:
            print(f"  Extracting {n_real_pos} positive samples...")
            
            # Get all positive samples from the original dataset
            df = orig_ds.df
            positive_samples = df[df['target'] == 1].copy()
            
            if len(positive_samples) < n_real_pos:
                print(f"    WARNING: Only {len(positive_samples)} positive samples available, requested {n_real_pos}")
                # Use all available positive samples
                selected_samples = positive_samples
            else:
                # Sample n_real_pos positive samples deterministically
                np.random.seed(seed)
                selected_indices = np.random.choice(
                    len(positive_samples), 
                    size=n_real_pos, 
                    replace=False
                )
                selected_samples = positive_samples.iloc[selected_indices].copy()
            
            # Save to CSV
            output_file = output_dir / f"llm_samples_{n_real_pos}.csv"
            selected_samples.to_csv(output_file, index=False)
            
            print(f"    Saved {len(selected_samples)} positive samples to {output_file}")
            
            # Print sample info
            print(f"    Sample distribution:")
            print(f"      - Total positive samples in dataset: {len(positive_samples)}")
            print(f"      - Selected samples: {len(selected_samples)}")
            if 'prompt_len' in selected_samples.columns:
                avg_len = selected_samples['prompt_len'].mean()
                print(f"      - Average prompt length: {avg_len:.1f}")
    
    print(f"\n✅ Extraction complete! CSV files saved to: {output_dir}")
    print(f"📁 Files created:")
    for n_real_pos in n_real_pos_values:
        print(f"   - llm_samples_{n_real_pos}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Extract positive samples from datasets for LLM upsampling"
    )
    parser.add_argument(
        "-c", "--config", 
        required=True, 
        help="Config name (e.g. 'spam_exp') or path to config YAML file"
    )
    
    args = parser.parse_args()
    config_arg = args.config
    
    # Expand short config name to full path if needed

    config_path = Path('configs') / Path(f"{config_arg}_config.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        extract_positive_samples_for_llm(config_path)
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 