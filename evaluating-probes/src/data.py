from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer

from src.logger import Logger
from src.activations import ActivationManager
import importlib

__all__ = [
    "Dataset",
    "get_available_datasets",
    "load_combined_classification_datasets",
]

from functools import lru_cache

@lru_cache(maxsize=1)
def get_main_csv_metadata() -> pd.DataFrame:  # type: ignore[override]
    path = Path("datasets/main.csv")
    if not path.exists():
        raise FileNotFoundError("datasets/main.csv not found. This file is required for dataset metadata.")
    df = pd.read_csv(path)
    # print("[DEBUG] main.csv 'number' column:", df['number'].tolist())
    df['number'] = df['number'].astype(int)
    # print("[DEBUG] main.csv index after set_index:", df['number'].tolist())
    return df.set_index("number")

class Dataset:
    """Dataset wrapper that lazily populates activation caches."""

    def __init__(
        self,
        dataset_name: str,
        *,
        model: HookedTransformer | None,
        device: str = "cuda:0",
        data_dir: Path = Path("datasets/cleaned"),
        cache_root: Path = Path("activation_cache"),
        seed: int = 42, # This is so important - how train and test sets persist across models/runs/etc.
    ):
        self.dataset_name = dataset_name
        self.model = model  # Ensure model is always set as an attribute
        self.device = device
        self.cache_root = cache_root
        self.seed = seed
        
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

        # Placeholder, updated before generating activations
        self.max_len = ( # We can safely divide by 2 because this is character length vs. token count, and avg > 2
            df["prompt_len"].max()//2 if "prompt_len" in df.columns else df["prompt"].str.len().max()//2
        )

        if "classification" in self.task_type:
            # Map minority class to 1, majority to 0 for binary classification
            value_counts = df["target"].astype(str).value_counts()
            if len(value_counts) == 2:
                # Binary: assign 1 to minority, 0 to majority
                classes_sorted = value_counts.index[::-1]  # minority first
                class_to_int = {classes_sorted[0]: 1, classes_sorted[1]: 0}
                df["target"] = df["target"].astype(str).map(class_to_int)
                uniq = np.array(list(classes_sorted))
            else:
                # Multiclass: use default factorize
                df["target"], uniq = pd.factorize(df["target"].astype(str))
            self.n_classes = len(np.unique(df["target"]))
        else:
            df["target"] = pd.to_numeric(df["target"], errors="coerce")
            self.n_classes = None

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

        try: 
            cache_dir = cache_root / model.cfg.model_name / dataset_name
            self.act_manager = ActivationManager(
                model=model,
                device=device,
                d_model=model.cfg.d_model,
                max_len=self.max_len,
                cache_dir=cache_dir,
            )
        except: 
            print("Not using dataset class to manage activations. Model is likely None.")

    def split_data(self, train_size: float = 0.75, val_size: float = 0.10, test_size: float = 0.15, seed: int = None):
        """
        Split the data into train, val, and test sets. Can be called multiple times to override splits.
        Default: 75% train, 10% val, 15% test.
        """
        if seed is None:
            seed = 42  # Default seed
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

    def get_activations_for_texts(self, texts: List[str], layer: int, component: str):
        """Get activations for an arbitrary list of texts."""
        if not hasattr(self, 'act_manager'):
            raise ValueError("No activation manager available")
        acts = self.act_manager.get_activations_for_texts(texts, layer, component)
        return acts

    # text getters (unchanged)
    def get_train_set(self):
        if self.X_train_text is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        return self.X_train_text, self.y_train

    def get_val_set(self):
        if self.X_val_text is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        return self.X_val_text, self.y_val

    def get_test_set(self):
        if self.X_test_text is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        return self.X_test_text, self.y_test

    # activation getters
    def get_train_set_activations(self, layer: int, component: str):
        if self.X_train_text is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        # Update max_len for this split
        self.max_len = max(len(p) for p in self.X_train_text) // 2
        acts = self.act_manager.get_activations_for_texts(self.X_train_text, layer, component)
        return acts, self.y_train

    def get_val_set_activations(self, layer: int, component: str):
        if self.X_val_text is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        # Update max_len for this split
        self.max_len = max(len(p) for p in self.X_val_text) // 2
        acts = self.act_manager.get_activations_for_texts(self.X_val_text, layer, component)
        return acts, self.y_val

    def get_test_set_activations(self, layer: int, component: str):
        if self.X_test_text is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        # Update max_len for this split
        self.max_len = max(len(p) for p in self.X_test_text) // 2
        acts = self.act_manager.get_activations_for_texts(self.X_test_text, layer, component)
        return acts, self.y_test
    
    def rebuild(self, class_counts: dict = None, class_percents: dict = None, total_samples: int = None, seed: int = 42) -> 'Dataset':
        """
        Returns a new Dataset object with resampled data according to the specified class_counts or class_percents.
        Args:
            class_counts: dict mapping class label to desired count, e.g. {0: 100, 1: 200}
            class_percents: dict mapping class label to desired percent, e.g. {0: 0.3, 1: 0.7}
            total_samples: total number of samples (used with class_percents)
            seed: random seed for reproducibility
        """
        import copy
        np.random.seed(seed)
        df = self.df.copy()
        if class_counts is not None:
            # Sample specified number from each class
            dfs = []
            for label, count in class_counts.items():
                class_df = df[df['target'] == label]
                if len(class_df) < count:
                    raise ValueError(f"Not enough samples for class {label}: requested {count}, available {len(class_df)}")
                dfs.append(class_df.sample(n=count, random_state=seed))
            new_df = pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
        elif class_percents is not None and total_samples is not None:
            # Compute number of samples for each class
            dfs = []
            for label, percent in class_percents.items():
                count = int(round(percent * total_samples))
                class_df = df[df['target'] == label]
                if len(class_df) < count:
                    raise ValueError(f"Not enough samples for class {label}: requested {count}, available {len(class_df)}")
                dfs.append(class_df.sample(n=count, random_state=seed))
            new_df = pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            raise ValueError("Must specify either class_counts or (class_percents and total_samples)")
        # Create a new Dataset instance with the new DataFrame
        new_dataset = copy.copy(self)
        new_dataset.df = new_df
        new_dataset.X = new_df["prompt"].astype(str).to_numpy()
        new_dataset.y = new_df["target"].to_numpy()
        new_dataset.X_train_text = None
        new_dataset.y_train = None
        new_dataset.X_val_text = None
        new_dataset.y_val = None
        new_dataset.X_test_text = None
        new_dataset.y_test = None
        new_dataset.train_indices = None
        new_dataset.val_indices = None
        new_dataset.test_indices = None
        return new_dataset

    @classmethod
    def from_dataframe(cls, df, *, dataset_name, model, device, cache_root, seed, task_type=None, n_classes=None, max_len=None, train_indices=None, val_indices=None, test_indices=None):
        """
        Construct a Dataset from a DataFrame, using the same model/device/etc. as the original.
        If train/val/test indices are provided, set all split attributes accordingly.
        """
        import copy
        obj = copy.copy(cls.__new__(cls))
        obj.dataset_name = dataset_name
        obj.df = df.copy()
        obj.model = model
        obj.device = device
        obj.max_len = max_len if max_len is not None else (df["prompt_len"].max() if "prompt_len" in df.columns else df["prompt"].str.len().max())
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
        
        # Rewriting ActivationManager
        try:
            cache_dir = cache_root / model.cfg.model_name / dataset_name
            obj.act_manager = ActivationManager(
                model=model,
                device=device,
                d_model=model.cfg.d_model,
                max_len=obj.max_len,
                cache_dir=cache_dir,
            )
        except:
            print("Using dataset class to fetch text, not manage activations.")
        return obj

    @staticmethod
    def build_imbalanced_train_balanced_eval(
        original_dataset,
        train_class_counts=None,
        train_class_percents=None,
        train_total_samples=None,
        val_size: float = 0.10,
        test_size: float = 0.15,
        seed: int = 42,
        llm_upsample: bool = False,
        llm_csv_path=None,
        num_real_pos: int = 5,
    ):
        """
        Returns a single Dataset object with precomputed splits:
        - test and val are both balanced 50/50 (as large as possible given split sizes and data)
        - train is constructed from the remaining data using the requested class balance
        - If llm_upsample is True, upsample positives in train split using LLM samples from llm_csv_path
        - Default split: 75% train, 10% val, 15% test
        - No overlap between splits
        """
        import copy
        import pandas as pd
        np.random.seed(seed)
        df = original_dataset.df.copy()
        y = df['target'].to_numpy()
        classes = np.unique(y)
        n_total = len(df)
        n_test = int(round(test_size * n_total))
        n_val = int(round(val_size * n_total))
        n_per_class_test = n_test // len(classes)
        n_per_class_val = n_val // len(classes)
        # Build test/val indices (real data only, 50/50)
        test_indices, val_indices, used_indices = [], [], set()
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            if len(cls_indices) < n_per_class_test + n_per_class_val:
                raise ValueError(f"Not enough samples for class {cls} to fill test+val splits.")
            test_indices.extend(cls_indices[:n_per_class_test])
            val_indices.extend(cls_indices[n_per_class_test:n_per_class_test+n_per_class_val])
            used_indices.update(cls_indices[:n_per_class_test+n_per_class_val])
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
        if llm_upsample:
            n_real_neg = train_class_counts.get(0, 0) if train_class_counts else 0
            n_pos = train_class_counts.get(1, 0) if train_class_counts else 0
            # Use as many real positives as available (should always be num_real_pos of them), rest from LLM
            n_real_pos = min(np.sum(y_avail == 1), num_real_pos)
            n_llm_pos = n_pos - n_real_pos
            real_neg = df.iloc[available_indices][df.iloc[available_indices]['target'] == 0].sample(n=n_real_neg, random_state=seed)
            real_pos = df.iloc[available_indices][df.iloc[available_indices]['target'] == 1].sample(n=n_real_pos, random_state=seed) if n_real_pos > 0 else pd.DataFrame(columns=df.columns)
            if llm_csv_path is None:
                raise ValueError("llm_csv_path must be provided when llm_upsample is True")
            llm_df = pd.read_csv(llm_csv_path)
            llm_pos = llm_df[llm_df['target'] == 1]
            if len(llm_pos) < n_llm_pos:
                raise ValueError(f"Not enough LLM positive samples: requested {n_llm_pos}, available {len(llm_pos)}")
            llm_pos = llm_pos.sample(n=n_llm_pos, random_state=seed)
            train_df = pd.concat([real_neg, real_pos, llm_pos]).sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            train_counts = get_counts(train_class_counts, train_class_percents, train_total_samples)
            for cls in train_counts:
                cls_avail_indices = available_indices[y_avail == cls]
                n_train = train_counts[cls]
                if len(cls_avail_indices) < n_train:
                    raise ValueError(f"Not enough samples for class {cls} in train split: requested {n_train}, available {len(cls_avail_indices)}")
                np.random.shuffle(cls_avail_indices)
                train_indices.extend(cls_avail_indices[:n_train])
            # Shuffle train indices
            np.random.shuffle(train_indices)
            train_df = df.iloc[train_indices]
        # Build val/test DataFrames
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
        # Build new Dataset object
        all_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        train_indices_new = np.arange(len(train_df))
        val_indices_new = np.arange(len(train_df), len(train_df) + len(val_df))
        test_indices_new = np.arange(len(train_df) + len(val_df), len(all_df))
        # Overlap checks
        train_set = set(train_indices_new)
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
            dataset_name=original_dataset.dataset_name + ('_llm_upsampled' if llm_upsample else ''),
            model=original_dataset.model,
            device=original_dataset.device,
            cache_root=original_dataset.cache_root,
            seed=seed,
            task_type=original_dataset.task_type,
            n_classes=original_dataset.n_classes,
            max_len=original_dataset.max_len,
            train_indices=train_indices_new,
            val_indices=val_indices_new,
            test_indices=test_indices_new,
        )

    def get_contrast_pairs(self, contrast_fn):
        """
        For each row in the dataset, apply contrast_fn(row) to get (original_row, contrast_row).
        Returns a list of (original_row, contrast_row) tuples.
        """
        pairs = []
        for _, row in self.df.iterrows():
            orig, contrast = contrast_fn(row)
            pairs.append((orig, contrast))
        return pairs

    def get_contrast_activations(self, contrast_fn, layer: int, component: str, split: str = "train"):
        """
        For each data point in the specified split, generate (original, contrast) prompts using contrast_fn,
        get activations for both using the activation cache, and return the subtracted activations and labels.
        Also updates self.max_len to the maximum prompt length in the contrast pairs.
        """
        if split == "train":
            if self.train_indices is None:
                raise ValueError("Data not split yet. Call split_data() first.")
            indices = self.train_indices
        elif split == "val":
            if self.val_indices is None:
                raise ValueError("Data not split yet. Call split_data() first.")
            indices = self.val_indices
        elif split == "test":
            if self.test_indices is None:
                raise ValueError("Data not split yet. Call split_data() first.")
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split: {split}")

        orig_prompts = []
        contrast_prompts = []
        labels = []
        all_lengths = []
        for idx in indices:
            row = self.df.iloc[idx]
            orig, contrast = contrast_fn(row)
            orig_prompts.append(orig['prompt'])
            contrast_prompts.append(contrast['prompt'])
            labels.append(orig['target'])  # Use original label
            all_lengths.append(len(orig['prompt']))
            all_lengths.append(len(contrast['prompt']))
        # Update max_len for activation extraction
        self.max_len = max(all_lengths) // 2  # match original logic (character length / 2)
        orig_acts = self.act_manager.get_activations_for_texts(orig_prompts, layer, component)
        contrast_acts = self.act_manager.get_activations_for_texts(contrast_prompts, layer, component)
        contrast_sub = orig_acts - contrast_acts
        return contrast_sub, np.array(labels)

    @staticmethod
    def load_contrast_fn(fn_name: str):
        """
        Dynamically import and return the contrast function from contrast_probing/methods.py by name.
        """
        methods = importlib.import_module("src.contrast_probing.methods")
        return getattr(methods, fn_name)

def get_available_datasets(data_dir: Path = Path("datasets/cleaned")) -> List[str]:
    return [f.stem for f in data_dir.glob("*.csv")]