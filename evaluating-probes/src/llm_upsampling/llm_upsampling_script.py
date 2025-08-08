#!/usr/bin/env python3
"""
Automatic LLM upsampling script using OpenAI API o4-mini.

This script implements the LLM upsampling procedure:
1. Extract parameters (seeds, num_real_samples, upsampling_factors) from config file
2. Extract positive samples using the same seed-based sampling as build_imbalanced_train_balanced_eval
3. For each seed, create llm_samples folder in seed_*/ directory
4. For each num_real_samples, generate samples in batches of 3 until reaching maximum upsampling factor
5. Save final datasets as samples_{n_real_samples}.csv in each seed's llm_samples folder
6. Include safety checks and state machine for handling insufficient samples
7. Support incremental upsampling - if existing files are found, continue from where left off
8. Preserve all existing samples and generate new ones in consistent batch sizes

Usage:
    python -m src.llm_upsampling.llm_upsampling_script -c spam_exp --api-key YOUR_API_KEY

The script automatically reads:
- seeds: from config['seeds']
- num_real_samples: from llm_upsampling_experiments or increasing_spam_fixed_total config
- upsampling_factors: from llm_upsampling_experiments config (defaults to [2,3,4,5])

Output structure:
    results/{run_name}/seed_{seed}/llm_samples/
    ├── samples_1.csv    # 1 real sample + LLM generated (5x total)
    ├── samples_2.csv    # 2 real samples + LLM generated (10x total)
    ├── samples_4.csv    # 4 real samples + LLM generated (20x total)
    ├── samples_5.csv    # 5 real samples + LLM generated (25x total)
    └── samples_10.csv   # 10 real samples + LLM generated (50x total)
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
        print(f"Initializing LLM client with model: {model}")
        print(f"API key length: {len(api_key)} characters")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        print(f"LLM client initialized successfully")
        
    def extract_csv_from_response(self, response: str, target_samples: int = None) -> Optional[pd.DataFrame]:
        """Extract new prompts from LLM response and create DataFrame."""
        try:
            # Use the entire response content
            content = response.strip()
            
            # Remove any markdown formatting
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Debug: print the content
            # print(f"    Raw content:")
            # print(f"    {content}")
            
            # Parse new prompts using "NEW PROMPT" separator
            prompts = []
            lines = content.split('\n')
            current_prompt = ""
            in_prompt = False
            
            for line in lines:
                line = line.strip()
                if line == "NEW PROMPT":
                    if current_prompt:
                        # Remove quotes if present
                        current_prompt = current_prompt.strip()
                        if current_prompt.startswith('"') and current_prompt.endswith('"'):
                            current_prompt = current_prompt[1:-1]
                        prompts.append(current_prompt)
                    current_prompt = ""
                    in_prompt = True
                elif in_prompt and line:
                    if current_prompt:
                        current_prompt += " " + line
                    else:
                        current_prompt = line
            
            # Add the last prompt if there is one
            if current_prompt:
                current_prompt = current_prompt.strip()
                if current_prompt.startswith('"') and current_prompt.endswith('"'):
                    current_prompt = current_prompt[1:-1]
                prompts.append(current_prompt)
            
            # Create DataFrame with the new prompts
            df = pd.DataFrame({
                'prompt': prompts,
                'target': 1,  # All samples are class 1
                'prompt_len': [len(prompt) for prompt in prompts]
            })
            
            print(f"    Extracted {len(prompts)} new prompts")
            print(f"    Final DataFrame shape: {df.shape}")
            # print(f"    Sample prompts: {df['prompt'].head(3).tolist()}")
            
            # Validate that we have the expected number of samples
            if target_samples is not None and len(df) != target_samples:
                print(f"    WARNING: Expected {target_samples} samples, got {len(df)}")
            
            # Validate that prompts are not empty or just whitespace
            if any(not prompt.strip() for prompt in prompts):
                print(f"    WARNING: Found empty prompts")
                return None
            
            return df
            
        except Exception as e:
            print(f"Failed to parse response: {e}")
            print(f"Response was: {response}")
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
        
        # Check for prompt_len column (should be added by extract_csv_from_response)
        if 'prompt_len' not in upsampled_df.columns:
            return False, "Missing prompt_len column"
        
        # Check for duplicates within the new samples (rows after original samples)
        original_count = len(original_df)
        new_samples_df = upsampled_df.iloc[original_count:]
        
        if len(new_samples_df) > 0:
            # Check for duplicates within new samples only
            new_prompts = new_samples_df['prompt'].str.lower().str.strip()
            duplicate_count = len(new_prompts) - len(new_prompts.unique())
            if duplicate_count > 0:
                return False, f"Found {duplicate_count} duplicate prompts within new samples"
            
            # Check for duplicates between new samples and original samples
            original_prompts = set(original_df['prompt'].str.lower().str.strip())
            new_prompts_set = set(new_prompts)
            duplicates_with_original = original_prompts.intersection(new_prompts_set)
            if duplicates_with_original:
                return False, f"Found {len(duplicates_with_original)} new samples that duplicate original samples"
        
        return True, "Validation passed"
    
    def validate_new_samples(self, original_df: pd.DataFrame, new_samples_df: pd.DataFrame, 
                           batch_size: int) -> Tuple[bool, str]:
        """Validate only the new samples meet requirements."""
        
        # Check that all new samples are class 1 (positive)
        if 'target' in new_samples_df.columns:
            if not all(new_samples_df['target'] == 1):
                return False, "Some new samples are not class 1"
        
        # Check for required columns
        required_cols = ['prompt', 'target']
        missing_cols = [col for col in required_cols if col not in new_samples_df.columns]
        if missing_cols:
            return False, f"Missing required columns in new samples: {missing_cols}"
        
        # Check for prompt_len column (should be added by extract_csv_from_response)
        if 'prompt_len' not in new_samples_df.columns:
            return False, "Missing prompt_len column in new samples"
        
        # Check for duplicates within new samples
        new_prompts = new_samples_df['prompt'].str.lower().str.strip()
        duplicate_count = len(new_prompts) - len(new_prompts.unique())
        if duplicate_count > 0:
            return False, f"Found {duplicate_count} duplicate prompts within new samples"
        
        # Check for duplicates between new samples and original samples
        original_prompts = set(original_df['prompt'].str.lower().str.strip())
        new_prompts_set = set(new_prompts)
        duplicates_with_original = original_prompts.intersection(new_prompts_set)
        if duplicates_with_original:
            return False, f"Found {len(duplicates_with_original)} new samples that duplicate original samples"
        
        return True, "New samples validation passed"
    

    
    def generate_batch_upsampling_prompt(self, original_df: pd.DataFrame, batch_size: int, 
                                       previous_samples: Optional[pd.DataFrame] = None) -> str:
        """Generate the upsampling prompt for a batch of samples."""
        
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
        
        prompt = f"""You are an expert at creating synthetic data samples for machine learning. I need you to add {batch_size} new unique samples to the existing dataset.

        Here is the original dataset, as well as previous samples generated:
        {original_csv}{previous_csv}

        Requirements:
        - Analyze the existing samples to understand the data and its contents
        - Perform non-trivial upsampling (not just replacement of keywords or simple variations)
        - Ensure the new samples are realistic
        - Ensure the new samples are completely unique and diverse in structure from all existing samples
        - Generate exactly {batch_size} new samples that are all class 1 (no class 0 samples)

        Output format:
        NEW PROMPT
        [your first actual sample text]
        NEW PROMPT
        [your second actual sample text]
        [continue for exactly {batch_size} samples]
        
        CRITICAL: Generate REAL sample text based on the existing data. Do NOT use placeholder text."""

        return prompt
    
    def request_upsampling_batch(self, original_df: pd.DataFrame, batch_size: int, 
                                previous_samples: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Request a batch of upsampling from the LLM with retry logic."""
        
        prompt = self.generate_batch_upsampling_prompt(original_df, batch_size, previous_samples)
        
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{self.max_retries} for batch of {batch_size} samples...")
                print(f"    Using model: {self.model}")
                print(f"    Prompt length: {len(prompt)} characters")
                print(f"    Original samples: {len(original_df)}")
                print(f"    Batch size: {batch_size}")
                print(f"    Previous samples: {len(previous_samples) if previous_samples is not None else 0}")
                # print(f"    Prompt preview: {prompt[:500]}...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at creating synthetic data for machine learning tasks. Always respond with the exact number of samples requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                print(f"    API call successful, response object: {type(response)}")
                print(f"    Response choices: {len(response.choices)}")
                
                if len(response.choices) == 0:
                    print(f"    ERROR: No choices in response")
                    continue
                
                response_text = response.choices[0].message.content
                print(f"    Received response ({len(response_text)} characters)")
                print(f"    Response preview: {response_text[:200]}...")
                
                # Extract new samples from response
                new_samples_df = self.extract_csv_from_response(response_text, batch_size)
                if new_samples_df is None:
                    print(f"    Failed to parse response, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                
                # Validate that we got the right number of new samples
                if len(new_samples_df) != batch_size:
                    print(f"    ❌ Got {len(new_samples_df)} new samples, expected {batch_size}")
                    if attempt < self.max_retries - 1:
                        print(f"    Retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        print(f"    Max retries reached, returning None")
                        return None
                
                # Combine original samples with new samples
                combined_df = pd.concat([original_df, new_samples_df], ignore_index=True)
                
                # Validate the new samples (not the combined dataset)
                is_valid, validation_msg = self.validate_new_samples(
                    original_df, new_samples_df, batch_size
                )
                
                if is_valid:
                    print(f"    ✅ Validation passed: {len(new_samples_df)} new samples generated")
                    return new_samples_df  # Return only the new samples
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

def extract_config_parameters(config_path: Path) -> Tuple[List[int], List[int], List[int], str]:
    """
    Extract LLM upsampling parameters from config file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Tuple of (seeds, num_real_samples, upsampling_factors, train_on_dataset)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract seeds from config
    seeds = config.get('seeds', [42])
    if isinstance(seeds, int):
        seeds = [seeds]
    
    # Find the first experiment with train_on dataset
    train_on = None
    for experiment in config['experiments']:
        if 'train_on' in experiment:
            train_on = experiment['train_on']
            break
    
    if train_on is None:
        raise ValueError("No train_on dataset found in config")
    
    # Extract LLM upsampling parameters from the LLM upsampling experiment
    num_real_samples = set()
    upsampling_factors = set()
    
    # Look specifically for the LLM upsampling experiment
    for experiment in config['experiments']:
        if 'llm-upsampling' in experiment.get('name'):
            rebuild_config = experiment.get('rebuild_config', {})
            if 'llm_upsampling_experiments' in rebuild_config:
                llm_configs = rebuild_config['llm_upsampling_experiments']
                for llm_config in llm_configs:
                    if 'llm_upsampling' in llm_config and llm_config['llm_upsampling']:
                        n_real_pos = llm_config.get('n_real_pos')
                        upsampling_factor = llm_config.get('upsampling_factor')
                        if n_real_pos is not None:
                            num_real_samples.add(n_real_pos)
                        if upsampling_factor is not None:
                            upsampling_factors.add(upsampling_factor)
                break
    
    # If no LLM config found, raise an error
    if not num_real_samples:
        raise ValueError("No LLM upsampling experiment found in config. Please uncomment the '3-spam-pred-auc-llm-upsampling' experiment.")
    
    # Convert to sorted lists
    num_real_samples = sorted(list(num_real_samples))
    upsampling_factors = sorted(list(upsampling_factors))
    
    # If still no upsampling factors found, use defaults
    if not upsampling_factors:
        upsampling_factors = [2, 3, 4, 5]
    
    print(f"Extracted from config:")
    print(f"  Seeds: {seeds}")
    print(f"  Num real samples: {num_real_samples}")
    print(f"  Upsampling factors: {upsampling_factors}")
    print(f"  Train on dataset: {train_on}")
    
    return seeds, num_real_samples, upsampling_factors, train_on

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

def run_llm_upsampling(config_path: Path, api_key: str):
    """
    Run the complete LLM upsampling procedure for all seeds.
    
    Args:
        config_path: Path to config file
        api_key: OpenAI API key
    """
    
    # Extract parameters from config
    seeds, num_real_samples, upsampling_factors, train_on = extract_config_parameters(config_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    run_name = config['run_name']
    
    # Initialize state machine
    state_machine = LLMUpsamplingStateMachine(api_key)
    
    # Process each seed
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Processing seed {seed}")
        print(f"{'='*80}")
        
        # Create seed-specific output directory
        seed_output_dir = Path('results') / run_name / f'seed_{seed}' / 'llm_samples'
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {seed_output_dir}")
        
        # Extract samples using same logic as build_imbalanced_train_balanced_eval
        print(f"Extracting samples using seed {seed}...")
        extracted_samples = extract_samples_for_upsampling(config_path, num_real_samples, seed)
        
        # Process each sample count
        for n_real_samples in num_real_samples:
            print(f"\n{'-'*60}")
            print(f"Processing n_real_samples = {n_real_samples}")
            print(f"{'-'*60}")
            
            # Check if samples file already exists
            output_filename = f"samples_{n_real_samples}.csv"
            output_path = seed_output_dir / output_filename
            
            # Load existing data if available
            existing_df = None
            if output_path.exists():
                print(f"📁 Loading existing samples file: {output_path}")
                existing_df = pd.read_csv(output_path)
                print(f"   Found {len(existing_df)} existing samples")
                
                # Check what upsampling factors we already have
                existing_samples = len(existing_df)
                original_samples = n_real_samples
                current_upsampling_factor = existing_samples // original_samples
                
                print(f"   Current upsampling factor: {current_upsampling_factor}x")
                
                # Check if we need to generate more samples
                max_needed_factor = max(upsampling_factors)
                max_needed_samples = max_needed_factor * n_real_samples
                
                if existing_samples >= max_needed_samples:
                    print(f"✅ Already have enough samples for {max_needed_factor}x upsampling")
                    print(f"   Skipping LLM generation for n_real_samples = {n_real_samples}")
                    continue
                else:
                    print(f"🔄 Need to generate {max_needed_samples - existing_samples} more samples")
                    print(f"   Current: {existing_samples}, Target: {max_needed_samples}")
            else:
                print(f"🆕 No existing samples file found, starting from scratch")
                current_upsampling_factor = 0
            
            original_df = extracted_samples[n_real_samples]
            
            # Initialize current dataset - use existing if available, otherwise start with original
            if existing_df is not None:
                current_upsampled_df = existing_df.copy()
                print(f"   Starting with existing {len(current_upsampled_df)} samples")
            else:
                current_upsampled_df = original_df.copy()
                print(f"   Starting with original {len(current_upsampled_df)} samples")
            
            # Track all generated samples for this n_real_samples
            all_generated_samples = {}
            
            # Calculate the maximum target samples needed
            max_target_samples = max(upsampling_factors) * n_real_samples
            print(f"Maximum target samples needed: {max_target_samples}")
            
            # Generate samples in batches of 3 until we reach the maximum target
            batch_size = 3
            all_new_samples = []
            
            while len(current_upsampled_df) < max_target_samples:
                samples_needed = max_target_samples - len(current_upsampled_df)
                current_batch_size = min(batch_size, samples_needed)
                
                print(f"\n--- Generating batch of {current_batch_size} samples ---")
                print(f"Current samples: {len(current_upsampled_df)}")
                print(f"Target samples: {max_target_samples}")
                print(f"Samples needed: {samples_needed}")
                
                # Request batch upsampling
                new_samples_df = state_machine.request_upsampling_batch(
                    original_df, current_batch_size, current_upsampled_df
                )
                
                if new_samples_df is None:
                    print(f"❌ Failed to generate batch of {current_batch_size} samples")
                    break
                
                # Add new samples to our collection
                all_new_samples.append(new_samples_df)
                current_upsampled_df = pd.concat([current_upsampled_df, new_samples_df], ignore_index=True)
                
                print(f"✅ Added {len(new_samples_df)} new samples, total: {len(current_upsampled_df)}")
            
            # Create the final upsampled dataset
            if all_new_samples:
                final_df = current_upsampled_df.copy()
                
                # Calculate what upsampling factors we achieved
                achieved_factors = []
                for factor in upsampling_factors:
                    target_for_factor = factor * n_real_samples
                    if len(final_df) >= target_for_factor:
                        achieved_factors.append(factor)
                
                print(f"✅ Achieved upsampling factors: {achieved_factors}")
                print(f"✅ Final dataset has {len(final_df)} samples")
                
                # Store the final dataset
                all_generated_samples = {max(achieved_factors): final_df}
            else:
                print(f"❌ No new samples generated")
                all_generated_samples = {}
            
            # Save the final cumulative dataset for this n_real_samples
            if all_generated_samples:
                # Get the final dataset
                max_factor = max(all_generated_samples.keys())
                final_df = all_generated_samples[max_factor]
                
                # Verify we have the complete dataset (original + all generated)
                print(f"📊 Final dataset breakdown:")
                print(f"   - Original samples: {len(original_df)}")
                print(f"   - Total samples in final dataset: {len(final_df)}")
                print(f"   - Generated samples: {len(final_df) - len(original_df)}")
                print(f"   - Highest upsampling factor: {max_factor}x")
                
                # Save as samples_{n_real_samples}.csv
                output_filename = f"samples_{n_real_samples}.csv"
                output_path = seed_output_dir / output_filename
                final_df.to_csv(output_path, index=False)
                
                print(f"✅ Saved complete dataset with {len(final_df)} samples to: {output_path}")
                print(f"   (Includes {len(original_df)} original + {len(final_df) - len(original_df)} generated samples)")
                
                # Check if we reached the maximum upsampling factor
                max_needed_factor = max(upsampling_factors)
                if max_factor >= max_needed_factor:
                    print(f"🎉 Successfully completed all upsampling factors up to {max_factor}x")
                else:
                    print(f"⚠️  Only completed up to {max_factor}x, but {max_needed_factor}x was requested")
            else:
                print(f"❌ No samples generated for n_real_samples={n_real_samples}")
        
        print(f"\nSeed {seed} complete! Files saved to: {seed_output_dir}")
    
    print(f"\n{'='*80}")
    print(f"LLM upsampling complete for all seeds!")
    print(f"Results saved to: results/{run_name}/seed_*/llm_samples/")
    print(f"Upsampling factors processed: {upsampling_factors}")
    print(f"{'='*80}")

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
            api_key=args.api_key
        )
    except Exception as e:
        print(f"❌ Error during LLM upsampling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 