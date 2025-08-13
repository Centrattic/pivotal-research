from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer

from src.logger import Logger
from src.activations import ActivationManager
import importlib

__all__ = [
    "Dataset",
    "get_available_datasets",
]

from functools import lru_cache

@lru_cache(maxsize=1)
def get_main_csv_metadata() -> pd.DataFrame:  # type: ignore[override]
    path = Path("datasets/main.csv")
    if not path.exists():
        raise FileNotFoundError("datasets/main.csv not found. This file is required for dataset metadata.")
    df = pd.read_csv(path)
    df['number'] = df['number'].astype(int)
    return df.set_index("number")

def get_model_d_model(model_name: str) -> int:
    """Get the d_model dimension for a given model name."""
    model_dims = {
        'google/gemma-2-9b': 3584,
        'meta-llama/Llama-3.3-70B-Instruct': 8192,
    }
    if model_name not in model_dims:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(model_dims.keys())}")
    return model_dims[model_name]


class Dataset:
    """Dataset wrapper that lazily populates activation caches."""

    def __init__(
        self,
        dataset_name: str,
        *,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_name: str = None,  # Alternative to passing the full model
        device: str = "cuda:0",
        data_dir: Path = Path("datasets/cleaned"),
        cache_root: Path = Path("activation_cache"),
        seed: int,
        only_test: bool = False,
    ):
        self.dataset_name = dataset_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.cache_root = cache_root
        self.seed = seed
        self.only_test = only_test
        
        meta = get_main_csv_metadata()
        # Robust lookup: try 'name' column, else try integer index, else error
        if 'name' in meta.columns and dataset_name in meta['name'].values:
            row = meta[meta['name'] == dataset_name].iloc[0]
            number = row.name  # index is the number
            save_name = row['save_name']
            meta_row = row
            csv_path = data_dir / f"{number}_{save_name}"
        else:
            try:
                idx = int(dataset_name.split("_", 1)[0])
                meta_row = meta.loc[idx]
                csv_path = data_dir / f"{dataset_name}.csv"
            except Exception as e:
                raise ValueError(f"Could not find dataset '{dataset_name}' in main.csv 'name' column or as integer index. Error: {e}")
        self.task_type = meta_row["Data type"].strip().lower()

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path).dropna(subset=["prompt", "target"])
        self.df = df

        # Calculate max_len from activation cache if available, otherwise set to None
        self.max_len = None  # Will be updated from activation cache if available

        # Binary classification only - map minority class to 1, majority to 0
        value_counts = df["target"].astype(str).value_counts()
        if len(value_counts) == 2:
            # Binary: assign 1 to minority, 0 to majority
            classes_sorted = value_counts.index[::-1]  # minority first
            class_to_int = {classes_sorted[0]: 1, classes_sorted[1]: 0}
            df["target"] = df["target"].astype(str).map(class_to_int)
            uniq = np.array(list(classes_sorted))
        else:
            # If not exactly 2 classes, warn and use factorize
            print(f"Warning: Expected 2 classes for binary classification, got {len(value_counts)}")
            df["target"], uniq = pd.factorize(df["target"].astype(str))
        self.n_classes = len(np.unique(df["target"]))

        # Store all data before splitting
        self.X = df["prompt"].astype(str).to_numpy()
        self.y = df["target"].to_numpy()
        
        # Initialize split attributes
        self.X_train_text = None
        self.y_train = None
        self.X_val_text = None
        self.y_val = None
        self.X_test_text = None
        self.y_test = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # If only_test is True, override splits
        if only_test:
            # Find all indices for each class
            classes, counts = np.unique(self.y, return_counts=True)
            if len(classes) != 2:
                raise ValueError("only_test=True requires exactly 2 classes for 50/50 split.")
            min_class = classes[np.argmin(counts)]
            maj_class = classes[np.argmax(counts)]
            min_indices = np.where(self.y == min_class)[0]
            maj_indices = np.where(self.y == maj_class)[0]
            n = len(min_indices)
            # Randomly sample n from majority class
            rng = np.random.RandomState(seed)
            maj_indices_sample = rng.choice(maj_indices, size=n, replace=False)
            test_indices = np.concatenate([min_indices, maj_indices_sample])
            rng.shuffle(test_indices)
            self.X_test_text = self.X[test_indices].tolist()
            self.y_test = self.y[test_indices]
            self.test_indices = test_indices
            self.X_train_text = None
            self.y_train = None
            self.X_val_text = None
            self.y_val = None
            self.train_indices = None
            self.val_indices = None
        
        # Initialize activation manager if model is provided
        self.act_manager = None
        if model is not None or model_name is not None:
            try: 
                cache_dir = cache_root / self.model_name / dataset_name
                print(f"Initializing ActivationManager with model: {self.model_name}, device: {device}, cache_dir: {cache_dir}")
                print(f"[DEBUG] model_name parameter: {model_name}")
                print(f"[DEBUG] self.model_name: {self.model_name}")
                print(f"[DEBUG] model parameter: {model}")
                print(f"[DEBUG] model type: {type(model)}")
                
                if model is not None and tokenizer is not None:
                    # Full model and tokenizer provided - use them
                    # Get d_model from model config
                    d_model = getattr(model.config, 'hidden_size', None)
                    if d_model is None:
                        # Try alternative attribute names
                        d_model = getattr(model.config, 'n_embd', None)  # GPT-2 style
                    if d_model is None:
                        raise ValueError(f"Could not determine d_model from model config: {model.config}")
                    
                    self.act_manager = ActivationManager(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        d_model=d_model,
                        cache_dir=cache_dir,
                    )
                else:
                    # Only model name provided - create readonly activation manager
                    # that can read existing caches but not create new ones
                    print(f"[DEBUG] Creating read-only activation manager for model_name={model_name}")
                    try:
                        self.act_manager = ActivationManager.create_readonly(
                            model_name=model_name,
                            d_model=get_model_d_model(model_name),
                            cache_dir=cache_dir,
                        )
                        print(f"[DEBUG] Successfully created read-only activation manager")
                    except Exception as e:
                        print(f"[DEBUG] Failed to create read-only activation manager: {e}")
                        raise
                    
                print(f"Successfully initialized ActivationManager")
            except Exception as e: 
                print(f"Failed to initialize activation manager: {e}")
                print(f"Model type: {type(model)}, Model name: {self.model_name}")
                print("Not using dataset class to manage activations.")
                self.act_manager = None
        else:
            print("No model or model_name provided, not initializing activation manager.")

    def split_data(self, seed: int, train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15):
        """
        Split the data into train, val, and test sets. Can be called multiple times to override splits.
        Default: 75% train, 10% val, 15% test.
        """
        if self.only_test:
            return
            
        if seed is None:
            raise ValueError("Seed must be provided")
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
        n = len(self.df)
        indices = np.arange(n)
        # First split off test
        trval_idx, te_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=self.y if self.n_classes else None, shuffle=True
        )
        # Now split train/val
        val_relative = val_size / (train_size + val_size)
        tr_idx, va_idx = train_test_split(
            trval_idx, test_size=val_relative, random_state=seed, stratify=self.y[trval_idx] if self.n_classes else None, shuffle=True
        )
        self.train_indices = tr_idx
        self.val_indices = va_idx
        self.test_indices = te_idx
        self.X_train_text = self.X[tr_idx].tolist()
        self.y_train = self.y[tr_idx]
        self.X_val_text = self.X[va_idx].tolist()
        self.y_val = self.y[va_idx]
        self.X_test_text = self.X[te_idx].tolist()
        self.y_test = self.y[te_idx]
        print(f"Split data: {len(self.X_train_text)} train, {len(self.X_val_text)} val, {len(self.X_test_text)} test")

    def get_activations_for_texts(self, texts: List[str], layer: int, component: str, activation_type: str = "full"):
        """Get activations for an arbitrary list of texts."""
        if self.act_manager is None:
            raise ValueError("No activation manager available. Model may not be loaded.")
        acts, masks = self.act_manager.get_activations_for_texts(texts, layer, component, activation_type)
        return acts, masks

    # Text getters
    def get_train_set(self):
        if self.X_train_text is None:
            raise ValueError("Data not split yet. Split data first.")
        return self.X_train_text, self.y_train

    def get_val_set(self):
        if self.X_val_text is None:
            raise ValueError("Data not split yet. Split data first.")
        return self.X_val_text, self.y_val

    def get_test_set(self):
        if self.X_test_text is None:
            raise ValueError("Data not split yet. Split data first.")
        return self.X_test_text, self.y_test

    def update_max_len_from_activations(self, layer: int, component: str):
        """Update max_len based on actual token lengths from activations."""
        if self.act_manager is not None:
            # The ActivationManager will compute its own max_len when needed
            # We can also update our own max_len for compatibility
            actual_max_len = self.act_manager.get_actual_max_len(layer, component)
            if actual_max_len is not None:
                self.max_len = actual_max_len

    # Activation getters
    def get_train_set_activations(self, layer: int, component: str, use_masks: bool = True, activation_type: str = "full"):
        if self.X_train_text is None:
            raise ValueError("Data not split yet. Split data first.")
        if self.act_manager is None:
            raise ValueError("No activation manager available. Model may not be loaded.")
        # Update max_len from actual activations if available
        self.update_max_len_from_activations(layer, component)
        acts, masks = self.act_manager.get_activations_for_texts(self.X_train_text, layer, component, activation_type)
        
        # Validate activations for numerical issues
        self._validate_activations(acts, "train")
        
        if use_masks and masks is not None:
            return acts, masks, self.y_train
        else:
            return acts, self.y_train

    def get_val_set_activations(self, layer: int, component: str, use_masks: bool = True, activation_type: str = "full"):
        if self.X_val_text is None:
            raise ValueError("Data not split yet. Split data first.")
        if self.act_manager is None:
            raise ValueError("No activation manager available. Model may not be loaded.")
        # Update max_len from actual activations if available
        self.update_max_len_from_activations(layer, component)
        acts, masks = self.act_manager.get_activations_for_texts(self.X_val_text, layer, component, activation_type)
        
        # Validate activations for numerical issues
        self._validate_activations(acts, "val")
        
        if use_masks and masks is not None:
            return acts, masks, self.y_val
        else:
            return acts, self.y_val

    def get_test_set_activations(self, layer: int, component: str, use_masks: bool = True, activation_type: str = "full"):
        if self.X_test_text is None:
            raise ValueError("Data not split yet. Split data first.")
        if self.act_manager is None:
            raise ValueError("No activation manager available. Model may not be loaded.")
        # Update max_len from actual activations if available
        self.update_max_len_from_activations(layer, component)
        acts, masks = self.act_manager.get_activations_for_texts(self.X_test_text, layer, component, activation_type)
        
        # Validate activations for numerical issues
        self._validate_activations(acts, "test")
        
        if use_masks and masks is not None:
            return acts, masks, self.y_test
        else:
            return acts, self.y_test
    
    def extract_all_activations(self, layer: int, component: str):
        """Extract activations for all texts in the dataset without requiring splits."""
        if self.act_manager is None:
            raise ValueError("No activation manager available. Model may not be loaded.")
        
        print(f"Extracting activations for all {len(self.X)} texts in dataset {self.dataset_name}")
        acts, masks = self.act_manager.get_activations_for_texts(self.X.tolist(), layer, component)
        print(f"Successfully extracted activations: shape={acts.shape}")
        return acts, masks

    def _validate_activations(self, acts, split_name):
        """Validate activations for numerical issues."""
        if acts is None:
            return
            
        acts_array = np.array(acts) if isinstance(acts, list) else acts
        
        # Check for infinity
        if np.any(np.isinf(acts_array)):
            inf_count = np.sum(np.isinf(acts_array))
            total_elements = acts_array.size
            print(f"[WARNING] Found {inf_count}/{total_elements} infinity values in {split_name} activations")
            print(f"[WARNING] Infinity locations: {np.where(np.isinf(acts_array))}")
            
        # Check for NaN
        if np.any(np.isnan(acts_array)):
            nan_count = np.sum(np.isnan(acts_array))
            total_elements = acts_array.size
            print(f"[WARNING] Found {nan_count}/{total_elements} NaN values in {split_name} activations")
            print(f"[WARNING] NaN locations: {np.where(np.isnan(acts_array))}")
            
        # Check for extremely large values
        max_val = np.max(np.abs(acts_array))
        if max_val > 1e10:
            print(f"[WARNING] Found extremely large values in {split_name} activations: max_abs={max_val}")
            
        # Check data type
        print(f"[DEBUG] {split_name} activations shape: {acts_array.shape}, dtype: {acts_array.dtype}")
        print(f"[DEBUG] {split_name} activations range: [{np.min(acts_array):.6f}, {np.max(acts_array):.6f}]")
    
    def clear_activation_cache(self):
        """Clear activation cache to free memory."""
        if self.act_manager is not None:
            self.act_manager.clear_activation_cache()

    @classmethod
    def from_dataframe(cls, df, *, dataset_name, model=None, tokenizer=None, model_name=None, device, cache_root, seed, task_type=None, n_classes=None, max_len=None, train_indices=None, val_indices=None, test_indices=None):
        """
        Construct a Dataset from a DataFrame, using the same model/device/etc. as the original.
        If train/val/test indices are provided, set all split attributes accordingly.
        """
        import copy
        obj = copy.copy(cls.__new__(cls))
        obj.dataset_name = dataset_name
        obj.df = df.copy()
        obj.model = model
        obj.tokenizer = tokenizer
        obj.model_name = model_name or (getattr(model, 'config', {}).get('name_or_path', None) if model is not None else None)
        obj.device = device
        obj.cache_root = cache_root
        obj.seed = seed
        obj.max_len = max_len if max_len is not None else None
        obj.task_type = task_type
        obj.n_classes = n_classes
        obj.X = df["prompt"].astype(str).to_numpy()
        obj.y = df["target"].to_numpy()
        obj.X_train_text = None
        obj.y_train = None
        obj.X_val_text = None
        obj.y_val = None
        obj.X_test_text = None
        obj.y_test = None
        obj.train_indices = None
        obj.val_indices = None
        obj.test_indices = None
        if train_indices is not None and val_indices is not None and test_indices is not None:
            obj.train_indices = np.array(train_indices)
            obj.val_indices = np.array(val_indices)
            obj.test_indices = np.array(test_indices)
            obj.X_train_text = obj.X[obj.train_indices].tolist()
            obj.y_train = obj.y[obj.train_indices]
            obj.X_val_text = obj.X[obj.val_indices].tolist()
            obj.y_val = obj.y[obj.val_indices]
            obj.X_test_text = obj.X[obj.test_indices].tolist()
            obj.y_test = obj.y[obj.test_indices]
        elif val_indices is not None and test_indices is not None:
            # Only val and test sets (no train set)
            obj.train_indices = None
            obj.val_indices = np.array(val_indices)
            obj.test_indices = np.array(test_indices)
            obj.X_train_text = None
            obj.y_train = None
            obj.X_val_text = obj.X[obj.val_indices].tolist()
            obj.y_val = obj.y[obj.val_indices]
            obj.X_test_text = obj.X[obj.test_indices].tolist()
            obj.y_test = obj.y[obj.test_indices]
        
        # Initialize ActivationManager
        obj.act_manager = None
        if (model is not None and tokenizer is not None) or model_name is not None:
            try:
                cache_dir = cache_root / obj.model_name / dataset_name
                if model is not None and tokenizer is not None:
                    # Get d_model from model config
                    d_model = getattr(model.config, 'hidden_size', None)
                    if d_model is None:
                        # Try alternative attribute names
                        d_model = getattr(model.config, 'n_embd', None)  # GPT-2 style
                    if d_model is None:
                        raise ValueError(f"Could not determine d_model from model config: {model.config}")
                    
                    obj.act_manager = ActivationManager(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        d_model=d_model,
                        cache_dir=cache_dir,
                    )
                else:
                    # Use read-only activation manager
                    obj.act_manager = ActivationManager.create_readonly(
                        model_name=model_name,
                        d_model=get_model_d_model(model_name),
                        cache_dir=cache_dir,
                    )
            except Exception as e:
                print(f"Failed to initialize activation manager in from_dataframe: {e}")
                print("Using dataset class to fetch text, not manage activations.")
                obj.act_manager = None
        else:
            print("No model or model_name in from_dataframe, not initializing activation manager.")
        return obj

    @staticmethod
    def build_llm_upsampled_dataset(
        original_dataset,
        seed: int,
        n_real_neg: int,
        n_real_pos: int,
        upsampling_factor: int,
        llm_csv_base_path: str,
        val_size: float = 0.10,
        test_size: float = 0.15,
        only_test: bool = False,
    ):
        if only_test:
            return original_dataset
        import copy
        import pandas as pd
        np.random.seed(seed)
        
        df = original_dataset.df.copy()
        y = df['target'].to_numpy()
        classes = np.sort(np.unique(y))
        
        # Calculate total samples needed for balanced test/val splits
        n_total = len(df)
        n_test = int(round(test_size * n_total))
        n_val = int(round(val_size * n_total))
        n_per_class_test = n_test // len(classes)
        n_per_class_val = n_val // len(classes)
        
        # Build test/val indices (balanced 50/50 from real data only)
        test_indices, val_indices, used_indices = [], [], set()
        insufficient_samples = False
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            rng = np.random.RandomState(seed)
            cls_indices_shuffled = cls_indices.copy()
            rng.shuffle(cls_indices_shuffled)
            if len(cls_indices_shuffled) < n_per_class_test + n_per_class_val:
                if only_test:
                    print(f"Not enough samples for class {cls} to fill test+val splits. Using all samples as test set.")
                    insufficient_samples = True
                    break
                else:
                    raise ValueError(f"Not enough samples for class {cls} to fill test+val splits.")
            test_indices.extend(cls_indices_shuffled[:n_per_class_test])
            val_indices.extend(cls_indices_shuffled[n_per_class_test:n_per_class_test+n_per_class_val])
            used_indices.update(cls_indices_shuffled[:n_per_class_test+n_per_class_val])
        
        # If insufficient samples and only_test is True, use all data as test set
        if insufficient_samples and only_test:
            # Use all samples as test set
            all_indices = np.arange(n_total)
            test_df = df.iloc[all_indices]
            
            # Calculate max_len from all possible data (only from original df since we're not using LLM data)
            all_lengths = df["prompt_len"].max() if "prompt_len" in df.columns else df["prompt"].str.len().max()
            max_len = all_lengths
            
            return Dataset.from_dataframe(
                test_df,
                dataset_name=original_dataset.dataset_name,  # Use original dataset name for shared cache
                model=original_dataset.model,
                model_name=original_dataset.model_name,
                device=original_dataset.device,
                cache_root=original_dataset.cache_root,
                seed=seed,
                task_type=original_dataset.task_type,
                n_classes=original_dataset.n_classes,
                max_len=max_len,
                train_indices=None,
                val_indices=None,
                test_indices=np.arange(len(test_df)),
                only_test=only_test,
            )
        
        # Remove test/val indices from available pool for train
        available_indices = np.array([i for i in range(n_total) if i not in used_indices])
        y_avail = y[available_indices]
        
        # Get real samples for training
        real_neg_indices = available_indices[y_avail == 0]
        real_pos_indices = available_indices[y_avail == 1]
        
        # Check if we have enough real samples
        if len(real_neg_indices) < n_real_neg:
            raise ValueError(f"Not enough real negative samples: requested {n_real_neg}, available {len(real_neg_indices)}")
        if len(real_pos_indices) < n_real_pos:
            raise ValueError(f"Not enough real positive samples: requested {n_real_pos}, available {len(real_pos_indices)}")
        
        # Sample real negatives and positives deterministically
        rng = np.random.RandomState(seed)
        real_neg_selected = rng.choice(real_neg_indices, size=n_real_neg, replace=False)
        real_pos_selected = rng.choice(real_pos_indices, size=n_real_pos, replace=False)
        
        # Calculate how many total samples we need
        n_llm_pos = n_real_pos * upsampling_factor - n_real_pos  # Total needed - real samples

        # The LLM samples are saved in seed-specific folders
        llm_csv_path = Path(llm_csv_base_path) / f"seed_{seed}" / "llm_samples" / f"samples_{n_real_pos}.csv"
        if not llm_csv_path.exists():
            raise FileNotFoundError(f"LLM samples file not found: {llm_csv_path}")
        
        llm_df = pd.read_csv(llm_csv_path)
        llm_pos = llm_df[llm_df['target'] == 1] # make sure this check works
        
        if len(llm_pos) < n_llm_pos:
            raise ValueError(f"Not enough LLM positive samples: requested {n_llm_pos}, available {len(llm_pos)}")
        
        # Take LLM samples in order (no shuffling)
        llm_pos_selected = llm_pos.head(n_llm_pos)
        
        # Combine real and LLM samples
        real_neg_df = df.iloc[real_neg_selected]
        real_pos_df = df.iloc[real_pos_selected]
        
        # Create train set: real negatives + real positives + LLM positives
        train_df = pd.concat([real_neg_df, real_pos_df, llm_pos_selected])
        # Shuffle train set deterministically
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Build val/test DataFrames from real data only
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
        
        # Build new Dataset object
        all_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        train_indices_new = np.arange(len(train_df))
        val_indices_new = np.arange(len(train_df), len(train_df) + len(val_df))
        test_indices_new = np.arange(len(train_df) + len(val_df), len(all_df))
        
        # Calculate max_len from all possible data
        all_lengths = max(
            df["prompt_len"].max() if "prompt_len" in df.columns else df["prompt"].str.len().max(),
            llm_df["prompt_len"].max() if "prompt_len" in llm_df.columns else llm_df["prompt"].str.len().max()
        )
        max_len = all_lengths
        
        # Use shared activation cache to avoid duplicating activations across LLM upsampling configurations
        # All LLM upsampling variants of the same base dataset will share the same activation cache
        return Dataset.from_dataframe(
            all_df,
            dataset_name=original_dataset.dataset_name,  # Use original dataset name for shared cache
            model=original_dataset.model,
            model_name=original_dataset.model_name,
            device=original_dataset.device,
            cache_root=original_dataset.cache_root,
            seed=seed,
            task_type=original_dataset.task_type,
            n_classes=original_dataset.n_classes,
            max_len=max_len,
            train_indices=train_indices_new,
            val_indices=val_indices_new,
            test_indices=test_indices_new,
        )

    @staticmethod
    def build_imbalanced_train_balanced_eval(
        original_dataset,
        seed: int,
        train_class_counts=None,
        train_class_percents=None,
        train_total_samples=None,
        val_size: float = 0.10,
        test_size: float = 0.15,
        only_test: bool = False,
    ):
        if only_test:
            return original_dataset
        import copy
        import pandas as pd
        np.random.seed(seed)
        df = original_dataset.df.copy()
        y = df['target'].to_numpy()
        # Sort classes to ensure deterministic order
        classes = np.sort(np.unique(y))
        n_total = len(df)
        n_test = int(round(test_size * n_total))
        n_val = int(round(val_size * n_total))
        n_per_class_test = n_test // len(classes)
        n_per_class_val = n_val // len(classes)
        
        # Calculate max_len from all possible data upfront
        all_lengths = df["prompt_len"].max() if "prompt_len" in df.columns else df["prompt"].str.len().max()
        max_len = all_lengths  # Use actual token lengths or character lengths directly
        
        # Build test/val indices (real data only, 50/50)
        test_indices, val_indices, used_indices = [], [], set()
        insufficient_samples = False
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            # Use consistent shuffling with the same seed for each class
            rng = np.random.RandomState(seed)
            cls_indices_shuffled = cls_indices.copy()
            rng.shuffle(cls_indices_shuffled)
            if len(cls_indices_shuffled) < n_per_class_test + n_per_class_val:
                if only_test:
                    print(f"Not enough samples for class {cls} to fill test+val splits. Using all samples as test set.")
                    insufficient_samples = True
                    break
                else:
                    raise ValueError(f"Not enough samples for class {cls} to fill test+val splits.")
            test_indices.extend(cls_indices_shuffled[:n_per_class_test])
            val_indices.extend(cls_indices_shuffled[n_per_class_test:n_per_class_test+n_per_class_val])
            used_indices.update(cls_indices_shuffled[:n_per_class_test+n_per_class_val])
        
        # If insufficient samples and only_test is True, use all data as test set
        if insufficient_samples and only_test:
            # Use all samples as test set
            all_indices = np.arange(n_total)
            test_df = df.iloc[all_indices]
            
            return Dataset.from_dataframe(
                test_df,
                dataset_name=original_dataset.dataset_name,
                model=original_dataset.model,
                model_name=original_dataset.model_name,
                device=original_dataset.device,
                cache_root=original_dataset.cache_root,
                seed=seed,
                task_type=original_dataset.task_type,
                n_classes=original_dataset.n_classes,
                max_len=max_len,
                train_indices=None,
                val_indices=None,
                test_indices=np.arange(len(test_df)),
            )
        
        # Check if we should create a train set
        create_train = (train_class_counts is not None or train_class_percents is not None or train_total_samples is not None)
        
        if create_train:
            # Remove test/val indices from available pool for train
            available_indices = np.array([i for i in range(n_total) if i not in used_indices])
            y_avail = y[available_indices]
            # Build train set
            train_indices = []
            def get_counts(class_counts, class_percents, total_samples):
                if class_counts is not None:
                    return class_counts
                elif class_percents is not None and total_samples is not None:
                    return {k: int(round(v * total_samples)) for k, v in class_percents.items()}
                else:
                    raise ValueError("Must specify either class_counts or (class_percents and total_samples)")
            
            train_counts = get_counts(train_class_counts, train_class_percents, train_total_samples)
            for cls in train_counts:
                cls_avail_indices = available_indices[y_avail == cls]
                n_train = train_counts[cls]
                if len(cls_avail_indices) < n_train:
                    raise ValueError(f"Not enough samples for class {cls} in train split: requested {n_train}, available {len(cls_avail_indices)}")
                # Use consistent shuffling with the same seed for each class
                rng = np.random.RandomState(seed)
                cls_avail_indices_shuffled = cls_avail_indices.copy()
                rng.shuffle(cls_avail_indices_shuffled)
                train_indices.extend(cls_avail_indices_shuffled[:n_train])
            # Shuffle train indices with consistent seed
            rng = np.random.RandomState(seed)
            train_indices_shuffled = train_indices.copy()
            rng.shuffle(train_indices_shuffled)
            train_df = df.iloc[train_indices_shuffled]
            
            # Build val/test DataFrames
            val_df = df.iloc[val_indices]
            test_df = df.iloc[test_indices]
            # Build new Dataset object with train, val, test
            all_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
            train_indices_new = np.arange(len(train_df))
            val_indices_new = np.arange(len(train_df), len(train_df) + len(val_df))
            test_indices_new = np.arange(len(train_df) + len(val_df), len(all_df))
        else:
            # Only create val/test sets
            val_df = df.iloc[val_indices]
            test_df = df.iloc[test_indices]
            # Build new Dataset object with only val and test
            all_df = pd.concat([val_df, test_df]).reset_index(drop=True)
            train_indices_new = None
            val_indices_new = np.arange(len(val_df))
            test_indices_new = np.arange(len(val_df), len(all_df))
        
        # Overlap checks
        train_set = set(train_indices_new) if train_indices_new is not None else set()
        val_set = set(val_indices_new)
        test_set = set(test_indices_new)
        overlap_tv = train_set & val_set
        overlap_tt = train_set & test_set
        overlap_vt = val_set & test_set
        if overlap_tv or overlap_tt or overlap_vt:
            print(f"WARNING: Overlap detected in splits! train/val: {len(overlap_tv)}, train/test: {len(overlap_tt)}, val/test: {len(overlap_vt)}")
        else:
            print("No overlap between train, val, and test sets.")
        
        return Dataset.from_dataframe(
            all_df,
            dataset_name=original_dataset.dataset_name,
            model=original_dataset.model,
            model_name=original_dataset.model_name,
            device=original_dataset.device,
            cache_root=original_dataset.cache_root,
            seed=seed,
            task_type=original_dataset.task_type,
            n_classes=original_dataset.n_classes,
            max_len=max_len,  # Use the precalculated max_len
            train_indices=train_indices_new,
            val_indices=val_indices_new,
            test_indices=test_indices_new,
        )

    def get_actual_max_len(self, layer: int, component: str) -> int:
        """Get the actual maximum token length from activations if available."""
        if self.act_manager is not None:
            return self.act_manager.get_actual_max_len(layer, component)
        return None

def get_available_datasets(data_dir: Path = Path("datasets/cleaned")) -> List[str]:
    return [f.stem for f in data_dir.glob("*.csv")]