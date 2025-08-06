#!/usr/bin/env python3
"""
Automatic LLM upsampling script using OpenAI API o4-mini.

This script implements the LLM upsampling procedure:
1. Extract positive samples using the same seed-based sampling as build_imbalanced_train_balanced_eval
2. For each num_real_samples in [1,2,4,5,10], create CSV files
3. Use o4-mini to upsample 2x, 3x, 4x, 5x consecutively
4. Include safety checks and state machine for handling insufficient samples

Usage:
    python -m src.llm_upsampling.llm_upsampling_script -c spam_exp --api-key YOUR_API_KEY
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import csv
import io
import time
import json
from pathlib import Path
import os
import sys
from typing import List, Dict, Optional, Tuple
import openai
from src.data import Dataset
from src.logger import Logger

class LLMUpsamplingStateMachine:
    """State machine for handling LLM upsampling with safety checks."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
    def extract_csv_from_response(self, response: str) -> Optional[pd.DataFrame]:
        """Extract CSV data from LLM response."""
        try:
            # Look for "Response below" marker
            if "Response below" in response:
                csv_start = response.find("Response below") + len("Response below")
                csv_content = response[csv_start:].strip()
            else:
                # Try to find CSV content directly
                csv_content = response.strip()
            
            # Remove any markdown formatting
            if csv_content.startswith("```csv"):
                csv_content = csv_content[6:]
            if csv_content.endswith("```"):
                csv_content = csv_content[:-3]
            
            csv_content = csv_content.strip()
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(csv_content))
            return df
            
        except Exception as e:
            print(f"Failed to parse CSV from response: {e}")
            return None
    
    def validate_upsampled_data(self, original_df: pd.DataFrame, upsampled_df: pd.DataFrame, 
                               target_samples: int, upsampling_factor: int) -> Tuple[bool, str]:
        """Validate upsampled data meets requirements."""
        if len(upsampled_df) < target_samples:
            return False, f"Not enough samples: got {len(upsampled_df)}, need {target_samples}"
        
        # Check that all samples are class 1 (positive)
        if 'target' in upsampled_df.columns:
            if not all(upsampled_df['target'] == 1):
                return False, "Some samples are not class 1"
        
        # Check for required columns
        required_cols = ['prompt', 'target']
        missing_cols = [col for col in required_cols if col not in upsampled_df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for duplicates with original data
        original_prompts = set(original_df['prompt'].str.lower().str.strip())
        upsampled_prompts = set(upsampled_df['prompt'].str.lower().str.strip())
        duplicates = original_prompts.intersection(upsampled_prompts)
        if duplicates:
            return False, f"Found {len(duplicates)} duplicate prompts with original data"
        
        return True, "Validation passed"
    
    def generate_upsampling_prompt(self, original_df: pd.DataFrame, upsampling_factor: int, 
                                 target_samples: int, previous_samples: Optional[pd.DataFrame] = None) -> str:
        """Generate the upsampling prompt for the LLM."""
        
        # Convert original data to CSV string
        csv_buffer = io.StringIO()
        original_df.to_csv(csv_buffer, index=False)
        original_csv = csv_buffer.getvalue()
        
        # Include previous samples if available
        previous_csv = ""
        if previous_samples is not None and len(previous_samples) > 0:
            csv_buffer = io.StringIO()
            previous_samples.to_csv(csv_buffer, index=False)
            previous_csv = f"\n\nPreviously generated samples:\n{csv_buffer.getvalue()}"
        
        prompt = f"""You are an expert at creating synthetic data samples for machine learning. I need you to upsample a dataset by analyzing the provided examples and generating new, unique samples that follow the same patterns.

Original dataset (CSV format):
{original_csv}{previous_csv}

Please upsample this dataset {upsampling_factor}x, by creating additional new and unique data samples that are 100% unique. The final dataset should have {target_samples} samples of class 1 (do not make any class 0 samples!). 

Important requirements:
1. Do not use any libraries or code - just generate the CSV data directly
2. Analyze the original samples to understand the data characteristics and patterns
3. Perform non-trivial upsampling (not just replacement of keywords or simple variations)
4. Ensure all new samples are class 1
5. Make sure all samples are completely unique from the original dataset
6. Maintain the same CSV format with 'prompt' and 'target' columns
7. Generate samples that are realistic and follow the same style/characteristics as the original data
8. **IMPORTANT**: If previously generated samples are provided above, include ALL of them in your output along with your new samples. The final dataset should contain the original samples + all previously generated samples + your new samples.

Please output your response in this exact format:
Response below
[CSV data here]

The CSV should have exactly {target_samples} rows total, which includes: original samples + all previously generated samples + your new additional samples."""

        return prompt
    
    def request_upsampling(self, original_df: pd.DataFrame, upsampling_factor: int, 
                          target_samples: int, previous_samples: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Request upsampling from the LLM with retry logic."""
        
        prompt = self.generate_upsampling_prompt(original_df, upsampling_factor, target_samples, previous_samples)
        
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{self.max_retries} for {upsampling_factor}x upsampling...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at creating synthetic data for machine learning tasks. Always respond with valid CSV data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                response_text = response.choices[0].message.content
                print(f"    Received response ({len(response_text)} characters)")
                
                # Extract CSV from response
                upsampled_df = self.extract_csv_from_response(response_text)
                if upsampled_df is None:
                    print(f"    Failed to parse CSV, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                
                # Validate the upsampled data
                is_valid, validation_msg = self.validate_upsampled_data(
                    original_df, upsampled_df, target_samples, upsampling_factor
                )
                
                if is_valid:
                    print(f"    ✅ Validation passed: {len(upsampled_df)} samples generated")
                    return upsampled_df
                else:
                    print(f"    ❌ Validation failed: {validation_msg}")
                    if attempt < self.max_retries - 1:
                        print(f"    Retrying...")
                        time.sleep(self.retry_delay)
                    else:
                        print(f"    Max retries reached, returning None")
                        return None
                        
            except Exception as e:
                print(f"    Error during API call: {e}")
                if attempt < self.max_retries - 1:
                    print(f"    Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"    Max retries reached")
                    return None
        
        return None

def extract_samples_for_upsampling(config_path: Path, num_real_samples: List[int], seed: int) -> Dict[int, pd.DataFrame]:
    """
    Extract positive samples using the same logic as build_imbalanced_train_balanced_eval.
    
    Args:
        config_path: Path to config file
        num_real_samples: List of sample counts to extract
        seed: Random seed for deterministic sampling
    
    Returns:
        Dictionary mapping sample count to DataFrame
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # Get all positive samples
    df = orig_ds.df
    positive_samples = df[df['target'] == 1].copy()
    
    print(f"Total positive samples in dataset: {len(positive_samples)}")
    
    # Extract samples for each count
    extracted_samples = {}
    
    for n_samples in num_real_samples:
        if len(positive_samples) < n_samples:
            print(f"WARNING: Only {len(positive_samples)} positive samples available, requested {n_samples}")
            # Use all available samples
            selected_samples = positive_samples.copy()
        else:
            # Use the same deterministic sampling as build_imbalanced_train_balanced_eval
            np.random.seed(seed)
            rng = np.random.RandomState(seed)
            positive_indices = np.where(df['target'] == 1)[0]
            positive_indices_shuffled = positive_indices.copy()
            rng.shuffle(positive_indices_shuffled)
            selected_indices = positive_indices_shuffled[:n_samples]
            selected_samples = df.iloc[selected_indices].copy()
        
        extracted_samples[n_samples] = selected_samples
        print(f"Extracted {len(selected_samples)} samples for n_real_samples={n_samples}")
    
    return extracted_samples

def run_llm_upsampling(config_path: Path, api_key: str, num_real_samples: List[int] = [1, 2, 4, 5, 10], 
                      upsampling_factors: List[int] = [2, 3, 4, 5], seed: int = 42):
    """
    Run the complete LLM upsampling procedure.
    
    Args:
        config_path: Path to config file
        api_key: OpenAI API key
        num_real_samples: List of real sample counts to process
        upsampling_factors: List of upsampling factors to apply
        seed: Random seed for deterministic sampling
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    run_name = config['run_name']
    
    # Create output directories
    base_output_dir = Path('results') / run_name / 'llm_upsampling'
    csv_output_dir = base_output_dir / 'input_csvs'
    llm_output_dir = base_output_dir / 'llm_generated'
    
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    llm_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories:")
    print(f"  Input CSVs: {csv_output_dir}")
    print(f"  LLM generated: {llm_output_dir}")
    
    # Extract samples using same logic as build_imbalanced_train_balanced_eval
    print(f"\nExtracting samples using seed {seed}...")
    extracted_samples = extract_samples_for_upsampling(config_path, num_real_samples, seed)
    
    # Initialize state machine
    state_machine = LLMUpsamplingStateMachine(api_key)
    
    # Process each sample count
    for n_real_samples in num_real_samples:
        print(f"\n{'='*60}")
        print(f"Processing n_real_samples = {n_real_samples}")
        print(f"{'='*60}")
        
        original_df = extracted_samples[n_real_samples]
        
        # Save original samples
        original_csv_path = csv_output_dir / f"original_samples_{n_real_samples}.csv"
        original_df.to_csv(original_csv_path, index=False)
        print(f"Saved original samples to: {original_csv_path}")
        
        # Track all generated samples for this n_real_samples
        all_generated_samples = {}
        previous_samples = None
        
        # Apply upsampling factors consecutively
        for upsampling_factor in upsampling_factors:
            print(f"\n--- {upsampling_factor}x upsampling ---")
            
            # Calculate target samples
            target_samples = n_real_samples * upsampling_factor
            print(f"Target: {target_samples} samples (original: {n_real_samples} × {upsampling_factor})")
            
            # Request upsampling
            upsampled_df = state_machine.request_upsampling(
                original_df, upsampling_factor, target_samples, previous_samples
            )
            
            if upsampled_df is not None:
                # Save upsampled data
                output_filename = f"llm_samples_{n_real_samples}_{upsampling_factor}x.csv"
                output_path = llm_output_dir / output_filename
                upsampled_df.to_csv(output_path, index=False)
                
                all_generated_samples[upsampling_factor] = upsampled_df
                previous_samples = upsampled_df  # Use for next iteration
                
                print(f"✅ Saved {len(upsampled_df)} samples to: {output_path}")
            else:
                print(f"❌ Failed to generate samples for {upsampling_factor}x upsampling")
                break
        
        # Save summary for this n_real_samples
        summary = {
            'n_real_samples': n_real_samples,
            'original_samples': len(original_df),
            'upsampling_results': {}
        }
        
        for factor in upsampling_factors:
            if factor in all_generated_samples:
                summary['upsampling_results'][factor] = {
                    'target_samples': n_real_samples * factor,
                    'generated_samples': len(all_generated_samples[factor]),
                    'success': True
                }
            else:
                summary['upsampling_results'][factor] = {
                    'target_samples': n_real_samples * factor,
                    'generated_samples': 0,
                    'success': False
                }
        
        summary_path = llm_output_dir / f"summary_{n_real_samples}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved summary to: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"LLM upsampling complete!")
    print(f"Results saved to: {base_output_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Run automatic LLM upsampling using OpenAI API"
    )
    parser.add_argument(
        "-c", "--config", 
        required=True, 
        help="Config name (e.g. 'spam_exp') or path to config YAML file"
    )
    parser.add_argument(
        "--api-key", 
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--num-real-samples",
        nargs="+",
        type=int,
        default=[1, 2, 4, 5, 10],
        help="List of real sample counts to process (default: 1 2 4 5 10)"
    )
    parser.add_argument(
        "--upsampling-factors",
        nargs="+", 
        type=int,
        default=[2, 3, 4, 5],
        help="List of upsampling factors to apply (default: 2 3 4 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)"
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
    
    try:
        run_llm_upsampling(
            config_path=config_path,
            api_key=args.api_key,
            num_real_samples=args.num_real_samples,
            upsampling_factors=args.upsampling_factors,
            seed=args.seed
        )
    except Exception as e:
        print(f"❌ Error during LLM upsampling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 