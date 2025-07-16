import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict
from functools import lru_cache

@lru_cache(maxsize=1)
def get_main_csv_metadata() -> pd.DataFrame:
    """Loads and caches the main.csv file for fast metadata lookup."""
    path = Path("datasets/main.csv")
    if not path.exists():
        raise FileNotFoundError("datasets/main.csv not found. This file is required for dataset metadata.")
    return pd.read_csv(path).set_index('number')

class Dataset:
    """
    Loads, cleans, and prepares a dataset for probing tasks.
    - Converts classification string labels to integers (with label_map).
    - Drops rows with missing prompts or targets.
    """
    def __init__(
        self, dataset_name: str, 
        data_dir: Path = Path("datasets/cleaned"), 
        test_size: float = 0.2, 
        seed: int = 42
    ):
        if dataset_name == "single_all":
            raise RuntimeError(
                "Dataset('single_all', ...) called directly! Use Dataset.from_combined(...) or load_combined_classification_datasets()."
            )
        self.dataset_name = dataset_name

        main_meta = get_main_csv_metadata()
        dataset_number = int(dataset_name.split('_')[0])
        self.metadata = main_meta.loc[dataset_number]
        self.task_type: str = (self.metadata['Data type'].strip()).lower()

        self.file_path = data_dir / f"{dataset_name}.csv"
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        self.df.dropna(subset=['prompt', 'target'], inplace=True)

        # If present, use prompt_len column for max_len, else use string length
        self.max_len: int = (
            self.df['prompt_len'].max()
            if 'prompt_len' in self.df.columns
            else self.df['prompt'].astype(str).str.len().max()
        )

        self.n_classes: Optional[int] = None
        self.label_map: Optional[Dict[int, str]] = None

        if "classification" in self.task_type:
            print(f"  - Note: Forcing integer encoding for '{self.dataset_name}'.")
            self.df['target'], uniques = pd.factorize(self.df['target'].astype(str))
            self.label_map = {i: label for i, label in enumerate(uniques)}
            self.df['target'] = self.df['target'].astype(int)
            self.n_classes = len(self.df['target'].unique())
        elif "continuous" in self.task_type or "regression" in self.task_type:
            self.df['target'] = pd.to_numeric(self.df['target'], errors='coerce')
            self.df.dropna(subset=['target'], inplace=True)
            self.df['target'] = self.df['target'].astype(float)

        # Check for empty dataset
        if self.df.shape[0] == 0:
            raise ValueError(
                f"Dataset '{self.dataset_name}' is empty after cleaning. No samples left to split."
            )

        self._perform_split(test_size, seed)

        # Ensure split labels are correct type
        if "classification" in self.task_type:
            self.y_train = self.y_train.astype(np.int32)
            self.y_test = self.y_test.astype(np.int32)
        elif "continuous" in self.task_type or "regression" in self.task_type:
            self.y_train = self.y_train.astype(np.float16)
            self.y_test = self.y_test.astype(np.float16)

        # Debug: print type and first few targets
        print(f"{self.dataset_name}: y_train task type: {self.task_type} dtype: {self.y_train.dtype}, sample: {self.y_train[:5]} max_len {self.max_len}")

    def _perform_split(self, test_size: float, seed: int):
        prompts_arr = self.df["prompt"].astype(str).to_numpy()
        labels_arr = self.df["target"].to_numpy()
        stratify_option = labels_arr if "classification" in self.task_type else None
        indices = np.arange(len(self.df))

        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=stratify_option
        )

        self.X_train_text: List[str] = prompts_arr[train_indices].tolist()
        self.y_train: np.ndarray = labels_arr[train_indices]
        self.X_test_text: List[str] = prompts_arr[test_indices].tolist()
        self.y_test: np.ndarray = labels_arr[test_indices]

    def get_train_set(self) -> Tuple[List[str], np.ndarray]:
        return self.X_train_text, self.y_train

    def get_test_set(self) -> Tuple[List[str], np.ndarray]:
        return self.X_test_text, self.y_test

    @classmethod
    def from_combined(
        cls, 
        X_train_text, y_train, 
        X_test_text, y_test, 
        max_len, n_classes, label_map=None
    ):
        obj = cls.__new__(cls)
        obj.dataset_name = "single_all"
        obj.task_type = "binary classification"
        obj.n_classes = n_classes
        obj.max_len = max_len
        obj.label_map = label_map
        obj.X_train_text = X_train_text
        obj.y_train = np.array(y_train).astype(np.int64)
        obj.X_test_text = X_test_text
        obj.y_test = np.array(y_test).astype(np.int64)
        obj.df = None
        obj.metadata = None
        return obj

def get_available_datasets(data_dir: Path = Path("datasets/cleaned")) -> List[str]:
    """Scans the cleaned data directory for available dataset names."""
    if not data_dir.exists():
        return []
    return [f.stem for f in data_dir.glob("*.csv")]

def load_combined_classification_datasets(seed: int) -> Dataset:
    """
    Loads all BINARY classification datasets, combines their train/test splits,
    and returns a single, unified Dataset object.
    """
    print("Combining all BINARY classification datasets for meta-probe...")
    all_datasets = get_available_datasets()
    combined_X_train, combined_y_train = [], []
    combined_X_test, combined_y_test = [], []
    max_len = 0
    included_datasets = []

    for name in all_datasets:
        try:
            data = Dataset(name, seed=seed)
            # Use case-insensitive matching and ensure binary classification
            if "classification" in data.task_type and data.n_classes == 2:
                xtr, ytr = data.get_train_set()
                xte, yte = data.get_test_set()
                combined_X_train.extend(xtr)
                combined_y_train.append(ytr)
                combined_X_test.extend(xte)
                combined_y_test.append(yte)
                if data.max_len > max_len:
                    max_len = data.max_len
                included_datasets.append(name)
        except Exception as e:
            print(f"  - Warning: Could not load or process dataset '{name}'. Skipping. Error: {e}")

    if not included_datasets:
        raise ValueError("No binary classification datasets found to create a combined 'single_all' dataset.")

    combined_data = Dataset.from_combined(
        X_train_text=combined_X_train,
        y_train=np.concatenate(combined_y_train),
        X_test_text=combined_X_test,
        y_test=np.concatenate(combined_y_test),
        max_len=min(max_len, 512),
        n_classes=2
    )
    print(f"Combined dataset created from {len(included_datasets)} binary datasets. Train size: {len(combined_data.X_train_text)}, Test size: {len(combined_data.X_test_text)}")
    return combined_data
