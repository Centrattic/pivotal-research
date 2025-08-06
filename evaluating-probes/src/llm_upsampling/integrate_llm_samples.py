#!/usr/bin/env python3
"""
Integration script to combine LLM-generated samples with the existing data pipeline.

This script takes the LLM-generated samples and creates the CSV files expected by
the build_llm_upsampled_dataset method, allowing for direct comparison with
non-upsampling results.

Usage:
    python -m src.llm_upsampling.integrate_llm_samples -c spam_exp
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from typing import Dict, List
from src.data import Dataset

def integrate_llm_samples(config_path: Path, llm_output_dir: Path, seed: int = 42):
    """
    Integrate LLM-generated samples with the existing data pipeline.
    
    Args:
        config_path: Path to config file
        llm_output_dir: Directory containing LLM-generated samples
        seed: Random seed for consistency
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    run_name = config['run_name']
    
    # Find the first experiment with train_on dataset
    train_on = None
    for experiment in config['experiments']:
        if 'train_on' in experiment:
            train_on = experiment['train_on']
            break
    
    if train_on is None:
        raise ValueError("No train_on dataset found in config")
    
    print(f"Loading dataset: {train_on}")
    
    # Load original dataset
    orig_ds = Dataset(train_on, model=None, device='cpu', seed=seed)
    
    # Create output directory for integrated samples
    integrated_output_dir = Path('results') / run_name / 'llm_upsampling' / 'integrated_csvs'
    integrated_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Integrated samples will be saved to: {integrated_output_dir}")
    
    # Find all LLM-generated CSV files
    llm_csv_files = list(llm_output_dir.glob("llm_samples_*.csv"))
    
    if not llm_csv_files:
        print(f"No LLM-generated CSV files found in {llm_output_dir}")
        return
    
    print(f"Found {len(llm_csv_files)} LLM-generated CSV files")
    
    # Process each LLM-generated file
    for llm_csv_file in llm_csv_files:
        print(f"\nProcessing: {llm_csv_file.name}")
        
        # Parse filename to extract n_real_samples and upsampling_factor
        # Expected format: llm_samples_{n_real_samples}_{upsampling_factor}x.csv
        filename_parts = llm_csv_file.stem.split('_')
        if len(filename_parts) < 4:
            print(f"  Skipping {llm_csv_file.name}: unexpected filename format")
            continue
        
        try:
            n_real_samples = int(filename_parts[2])
            upsampling_factor = int(filename_parts[3].replace('x', ''))
        except ValueError:
            print(f"  Skipping {llm_csv_file.name}: could not parse n_real_samples or upsampling_factor")
            continue
        
        print(f"  n_real_samples: {n_real_samples}, upsampling_factor: {upsampling_factor}")
        
        # Load LLM-generated samples
        llm_df = pd.read_csv(llm_csv_file)
        print(f"  Loaded {len(llm_df)} LLM-generated samples")
        
        # Validate LLM samples
        if 'prompt' not in llm_df.columns or 'target' not in llm_df.columns:
            print(f"  Skipping {llm_csv_file.name}: missing required columns")
            continue
        
        # Ensure all samples are class 1
        if not all(llm_df['target'] == 1):
            print(f"  Warning: Some samples are not class 1 in {llm_csv_file.name}")
            # Filter to only class 1 samples
            llm_df = llm_df[llm_df['target'] == 1].copy()
            print(f"  Filtered to {len(llm_df)} class 1 samples")
        
        # Create the integrated CSV file expected by build_llm_upsampled_dataset
        # This should contain all LLM-generated samples for this n_real_samples
        integrated_filename = f"llm_samples_{n_real_samples}.csv"
        integrated_path = integrated_output_dir / integrated_filename
        
        # If this is the first upsampling factor for this n_real_samples, create new file
        # Otherwise, append to existing file
        if integrated_path.exists() and upsampling_factor > 2:
            # Append new samples to existing file
            existing_df = pd.read_csv(integrated_path)
            print(f"  Appending {len(llm_df)} samples to existing {len(existing_df)} samples")
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, llm_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['prompt'], keep='first')
            print(f"  After deduplication: {len(combined_df)} total samples")
            
            combined_df.to_csv(integrated_path, index=False)
        else:
            # Create new file
            llm_df.to_csv(integrated_path, index=False)
            print(f"  Created new file with {len(llm_df)} samples")
        
        print(f"  ✅ Saved integrated samples to: {integrated_path}")
    
    print(f"\n{'='*60}")
    print(f"Integration complete!")
    print(f"Integrated CSV files saved to: {integrated_output_dir}")
    print(f"{'='*60}")
    
    # Print summary of available integrated files
    integrated_files = list(integrated_output_dir.glob("llm_samples_*.csv"))
    if integrated_files:
        print(f"\nAvailable integrated files:")
        for file in sorted(integrated_files):
            df = pd.read_csv(file)
            print(f"  {file.name}: {len(df)} samples")
    
    return integrated_output_dir

def create_comparison_config(config_path: Path, integrated_output_dir: Path):
    """
    Create a modified config file that includes LLM upsampling experiments
    for comparison with non-upsampling results.
    
    Args:
        config_path: Path to original config file
        integrated_output_dir: Directory containing integrated LLM samples
    """
    
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find integrated CSV files
    integrated_files = list(integrated_output_dir.glob("llm_samples_*.csv"))
    
    if not integrated_files:
        print("No integrated CSV files found, skipping config creation")
        return
    
    # Extract n_real_samples values
    n_real_samples_values = []
    for file in integrated_files:
        filename_parts = file.stem.split('_')
        if len(filename_parts) >= 3:
            try:
                n_real_samples = int(filename_parts[2])
                n_real_samples_values.append(n_real_samples)
            except ValueError:
                continue
    
    n_real_samples_values = sorted(list(set(n_real_samples_values)))
    print(f"Found n_real_samples values: {n_real_samples_values}")
    
    # Create LLM upsampling experiments configuration
    llm_upsampling_experiments = []
    
    for n_real_samples in n_real_samples_values:
        for upsampling_factor in [1, 2, 3, 4, 5]:
            # Skip 1x upsampling (no LLM samples)
            if upsampling_factor == 1:
                continue
            
            llm_upsampling_experiments.append({
                'llm_upsampling': True,
                'n_real_neg': 3000,  # Fixed negative samples
                'n_real_pos': n_real_samples,
                'upsampling_factor': upsampling_factor,
                'seed': 42
            })
    
    # Create new experiment configuration
    new_experiment = {
        'name': '3-spam-pred-auc-llm-upsampling',
        'metric': 'auc',
        'class_names': {'0': 'Ham', '1': 'Spam'},
        'train_on': '94_better_spam',
        'evaluate_on': ['94_better_spam'],
        'score': ['all'],
        'rebuild_config': {
            'llm_upsampling_experiments': llm_upsampling_experiments
        }
    }
    
    # Add to config
    config['experiments'].append(new_experiment)
    
    # Save modified config
    output_config_path = config_path.parent / f"{config_path.stem}_with_llm_upsampling.yaml"
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Created comparison config: {output_config_path}")
    print(f"Added {len(llm_upsampling_experiments)} LLM upsampling experiments")

def main():
    parser = argparse.ArgumentParser(
        description="Integrate LLM-generated samples with existing data pipeline"
    )
    parser.add_argument(
        "-c", "--config", 
        required=True, 
        help="Config name (e.g. 'spam_exp') or path to config YAML file"
    )
    parser.add_argument(
        "--llm-output-dir",
        type=Path,
        help="Directory containing LLM-generated samples (auto-detected if not specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for consistency (default: 42)"
    )
    parser.add_argument(
        "--create-comparison-config",
        action="store_true",
        help="Create a modified config file for LLM upsampling experiments"
    )
    
    args = parser.parse_args()
    
    # Expand short config name to full path if needed
    if not args.config.endswith('.yaml') and not args.config.endswith('.yml'):
        config_path = Path('configs') / f"{args.config}_config.yaml"
    else:
        config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    # Auto-detect LLM output directory if not specified
    if args.llm_output_dir is None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        run_name = config['run_name']
        args.llm_output_dir = Path('results') / run_name / 'llm_upsampling' / 'llm_generated'
    
    if not args.llm_output_dir.exists():
        print(f"❌ LLM output directory not found: {args.llm_output_dir}")
        print("Please run the LLM upsampling script first.")
        sys.exit(1)
    
    try:
        # Integrate LLM samples
        integrated_output_dir = integrate_llm_samples(
            config_path=config_path,
            llm_output_dir=args.llm_output_dir,
            seed=args.seed
        )
        
        # Create comparison config if requested
        if args.create_comparison_config and integrated_output_dir:
            create_comparison_config(config_path, integrated_output_dir)
            
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 