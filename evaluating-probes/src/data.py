# src/data.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
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
    A comprehensive data handler for a single dataset. It loads the data,
    its metadata, and determines properties like max_len, task_type, and n_classes.
    """
    def __init__(self, dataset_name: str, data_dir: Path = Path("datasets/cleaned"), test_size: float = 0.3, seed: int = 42):
        self.dataset_name = dataset_name
        
        # Load metadata from main.csv
        main_meta = get_main_csv_metadata()
        dataset_number = int(dataset_name.split('_')[0])
        self.metadata = main_meta.loc[dataset_number]
        self.task_type: str = self.metadata['Data type'].strip()

        # Load cleaned data
        self.file_path = data_dir / f"{dataset_name}.csv"
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        self.df.dropna(subset=['prompt', 'target'], inplace=True)

        # Determine dataset-specific properties
        calculated_max_len = self.df['prompt_len'].max() if 'prompt_len' in self.df.columns else self.df['prompt'].str.len().max()
        self.max_len: int = calculated_max_len

        self.n_classes: Optional[int] = None
        if "Classification" in self.task_type:
            self.n_classes = len(self.df['target'].unique())

        self._perform_split(test_size, seed)

    def _perform_split(self, test_size: float, seed: int):
        prompts_arr = self.df["prompt"].astype(str).to_numpy()
        labels_arr = self.df["target"].to_numpy()

        stratify_option = None
        if "Classification" in self.task_type:
            if len(np.unique(labels_arr)) < len(labels_arr):
                unique_labels, counts = np.unique(labels_arr, return_counts=True)
                if all(counts >= 2):
                     stratify_option = labels_arr
        
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

def get_available_datasets(data_dir: Path = Path("datasets/cleaned")) -> List[str]:
    """Scans the cleaned data directory for available dataset names."""
    if not data_dir.exists(): return []
    return [f.stem for f in data_dir.glob("*.csv")]