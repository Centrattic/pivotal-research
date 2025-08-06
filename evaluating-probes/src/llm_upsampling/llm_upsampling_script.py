#!/usr/bin/env python3
"""
Automatic LLM upsampling script using OpenAI API o4-mini.

This script implements the LLM upsampling procedure:
1. Extract parameters (seeds, num_real_samples, upsampling_factors) from config file
2. Extract positive samples using the same seed-based sampling as build_imbalanced_train_balanced_eval
3. For each seed, create llm_samples folder in seed_*/ directory
4. For each num_real_samples, generate cumulative upsampling (2x, 3x, 4x, 5x)
5. Save final datasets as samples_{n_real_samples}.csv in each seed's llm_samples folder
6. Include safety checks and state machine for handling insufficient samples

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
        
        # Calculate how many new samples we need
        current_total = len(original_df) + (len(previous_samples) if previous_samples is not None else 0)
        new_samples_needed = target_samples - current_total
        
        prompt = f"""You are an expert at creating synthetic data samples for machine learning. I need you to upsample a dataset by analyzing the provided examples and generating EXACTLY {new_samples_needed} new unique samples based on the original data.

        Here is the original dataset, as well as previous samples generated:
        {original_csv}{previous_csv}

        Requirements:
        - Analyze the original samples to understand the data and its contents
        - Perform non-trivial upsampling (not just replacement of keywords or simple variations)
        - Generate exactly {new_samples_needed} new samples that are all class 1 (no class 0 samples)
        - Ensure the new samples are realistic, while being completely unique and diverse in structure from the original samples

        Output format:
        NEW PROMPT
        [your first actual sample text]
        NEW PROMPT
        [your second actual sample text]
        [continue for exactly {new_samples_needed} samples]
        
        CRITICAL: Generate REAL sample text based on the original data. Do NOT use placeholder text or example text."""

        return prompt
    
    def request_upsampling(self, original_df: pd.DataFrame, upsampling_factor: int, 
                          target_samples: int, previous_samples: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Request upsampling from the LLM with retry logic."""
        
        prompt = self.generate_upsampling_prompt(original_df, upsampling_factor, target_samples, previous_samples)
        
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{self.max_retries} for {upsampling_factor}x upsampling...")
                print(f"    Using model: {self.model}")
                print(f"    Prompt length: {len(prompt)} characters")
                print(f"    Original samples: {len(original_df)}")
                print(f"    Target samples: {target_samples}")
                print(f"    New samples needed: {target_samples - len(original_df)}")
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
                new_samples_df = self.extract_csv_from_response(response_text, target_samples - len(original_df))
                if new_samples_df is None:
                    print(f"    Failed to parse response, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                
                # Combine original samples with new samples
                combined_df = pd.concat([original_df, new_samples_df], ignore_index=True)
                
                # Validate the combined dataset
                is_valid, validation_msg = self.validate_upsampled_data(
                    original_df, combined_df, target_samples, upsampling_factor
                )
                
                if is_valid:
                    print(f"    ✅ Validation passed: {len(new_samples_df)} new samples generated, {len(combined_df)} total")
                    return new_samples_df  # Return only the new samples
                else:
                    print(f"    ❌ Validation failed: {validation_msg}")
                    
                    # Check if we need more samples (incremental generation)
                    if "Not enough samples" in validation_msg:
                        current_total = len(combined_df)
                        missing_samples = target_samples - current_total
                        print(f"    🔄 Need {missing_samples} more samples, requesting incremental generation...")
                        
                        # Generate prompt for missing samples only
                        incremental_prompt = self.generate_upsampling_prompt(
                            original_df, upsampling_factor, target_samples, new_samples_df
                        )
                        
                        # Make another API call for missing samples
                        try:
                            incremental_response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": "You are an expert at creating synthetic data for machine learning tasks. Always respond with the exact number of samples requested."},
                                    {"role": "user", "content": incremental_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=4000
                            )
                            
                            incremental_response_text = incremental_response.choices[0].message.content
                            additional_samples_df = self.extract_csv_from_response(incremental_response_text, missing_samples)
                            
                            if additional_samples_df is not None and len(additional_samples_df) > 0:
                                # Combine with previous samples
                                final_combined_df = pd.concat([combined_df, additional_samples_df], ignore_index=True)
                                
                                # Validate again
                                final_is_valid, final_validation_msg = self.validate_upsampled_data(
                                    original_df, final_combined_df, target_samples, upsampling_factor
                                )
                                
                                if final_is_valid:
                                    print(f"    ✅ Incremental generation successful: {len(additional_samples_df)} additional samples, {len(final_combined_df)} total")
                                    return pd.concat([new_samples_df, additional_samples_df], ignore_index=True)
                                else:
                                    print(f"    ❌ Incremental generation validation failed: {final_validation_msg}")
                        
                        except Exception as e:
                            print(f"    ❌ Incremental generation failed: {e}")
                    
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
            
            if output_path.exists():
                print(f"✅ Samples file already exists: {output_path}")
                print(f"   Skipping LLM generation for n_real_samples = {n_real_samples}")
                continue
            
            original_df = extracted_samples[n_real_samples]
            
            # Track all generated samples for this n_real_samples
            all_generated_samples = {}
            current_upsampled_df = original_df.copy()  # Start with original samples
            
            # Apply upsampling factors consecutively
            for upsampling_factor in upsampling_factors:
                print(f"\n--- {upsampling_factor}x upsampling ---")
                
                # Calculate target samples (total samples needed)
                target_samples = upsampling_factor * n_real_samples
                print(f"Target: {target_samples} samples total")
                
                if upsampling_factor == 1:
                    # 1x upsampling: just use original samples
                    upsampled_df = original_df.copy()
                    print(f"✅ Using original {len(upsampled_df)} samples for 1x upsampling")
                else:
                    # Calculate how many additional samples we need
                    additional_samples_needed = target_samples - len(current_upsampled_df)
                    print(f"Need {additional_samples_needed} additional samples from LLM")
                    
                    # Request upsampling - pass current upsampled dataset as previous samples
                    new_samples_df = state_machine.request_upsampling(
                        original_df, upsampling_factor, target_samples, current_upsampled_df
                    )
                    
                    if new_samples_df is None:
                        print(f"❌ Failed to generate samples for {upsampling_factor}x upsampling")
                        break
                    
                    # Combine current upsampled samples with new samples
                    upsampled_df = pd.concat([current_upsampled_df, new_samples_df], ignore_index=True)
                
                all_generated_samples[upsampling_factor] = upsampled_df
                current_upsampled_df = upsampled_df.copy()  # Update for next iteration
                print(f"✅ We have {len(upsampled_df)} samples for {upsampling_factor}x upsampling")
            
            # Save the final cumulative dataset for this n_real_samples
            if all_generated_samples:
                # Get the highest upsampling factor that succeeded
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
            else:
                print(f"❌ No samples generated for n_real_samples={n_real_samples}")
        
        print(f"\nSeed {seed} complete! Files saved to: {seed_output_dir}")
    
    print(f"\n{'='*80}")
    print(f"LLM upsampling complete for all seeds!")
    print(f"Results saved to: results/{run_name}/seed_*/llm_samples/")
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