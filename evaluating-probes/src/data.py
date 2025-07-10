# src/data.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class DataLoader:
    """
    Handles loading a cleaned dataset and splitting it into prompts and labels.
    """
    def __init__(self, dataset_name: str, data_dir: Path = Path("datasets/cleaned")): # using absolute path, so define from room in evaluating_probes/
        self.dataset_name = dataset_name
        self.file_path = data_dir / f"{dataset_name}.csv"
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.file_path}")
            
        self.df = pd.read_csv(self.file_path)

        # Validate required columns
        required_cols = ["prompt", "target"]
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Dataset {dataset_name} must contain 'prompt' and 'target' columns.")

    @property
    def prompts(self) -> List[str]:
        """Returns the list of prompts."""
        return self.df["prompt"].astype(str).tolist()

    @property
    def labels(self) -> np.ndarray:
        """Returns the numpy array of labels."""
        return self.df["target"].to_numpy()

    def split_text_and_labels(
        self, test_size: float = 0.3, seed: int = 42
    ) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
        """
        Splits the data into training and testing sets with stratification.

        Returns:
            A tuple containing (X_train_text, y_train, X_test_text, y_test).
        """
        prompts_arr = self.df["prompt"].astype(str).to_numpy()
        labels_arr = self.labels

        # Use stratification only if there are enough samples per class
        stratify_option = None
        unique_labels, counts = np.unique(labels_arr, return_counts=True)
        if all(counts >= 2): # Stratify only if each class has at least 2 members
             stratify_option = labels_arr
        else:
            print(f"Note: Cannot stratify dataset '{self.dataset_name}' due to small class sizes. Splitting without stratification.")

        indices = np.arange(len(self.df))
        
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=seed, 
            stratify=stratify_option
        )
        
        X_train_text = prompts_arr[train_indices].tolist()
        y_train = labels_arr[train_indices]
        
        X_test_text = prompts_arr[test_indices].tolist()
        y_test = labels_arr[test_indices]
        
        return X_train_text, y_train, X_test_text, y_test


def get_available_datasets(data_dir: Path = Path("datasets/cleaned")) -> List[str]: 
    """
    Scans the cleaned data directory and returns a list of all available dataset names.
    This is used when the config specifies `datasets: ["all"]`.
    """
    if not data_dir.exists():
        return []
    
    return [f.stem for f in data_dir.glob("*.csv")]