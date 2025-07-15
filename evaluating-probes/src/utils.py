
import numpy as np
import json
from typing import List
from src.logger import Logger
from pathlib import Path

def should_skip_dataset(dataset_name, data, logger=None):
    """ This defines conditions for datasets we should always skip. We have separate conditions for skipping a dataset for evaluation if you trained on it. """
    # SKIP: Max length
    if hasattr(data, "max_len") and data.max_len > 512:
        if logger: logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Max length ({data.max_len}) exceeds 512.")
        return True
    # SKIP: Continuous
    if hasattr(data, "task_type") and "continuous" in data.task_type.strip().lower():
        if logger: logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': Continuous data is not supported.")
        return True
    # SKIP: Any class has < 2 samples (classification only)
    if hasattr(data, "task_type") and "classification" in data.task_type.strip().lower():
        # Use y_train/y_test if available, else fallback to .df
        y = None
        if hasattr(data, 'y_train') and hasattr(data, 'y_test'):
            y = np.concatenate([data.y_train, data.y_test])
        elif hasattr(data, 'df'):
            y = np.array(getattr(data.df, 'target', []))
        if y is not None and len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            if len(counts) == 0 or counts.min() < 2:
                if logger: logger.log(f"  - ⏭️  INVALID Dataset '{dataset_name}': At least one class has <2 samples (counts: {dict(zip(unique, counts))}).")
                return True
    return False


def dump_loss_history(losses: List[float], out_path: Path, logger: Logger | None = None):
     """
     Write `[loss0, loss1, ...]` as pretty-printed JSON.
     """
     with open(out_path, "w") as fp:
         json.dump(losses, fp, indent=2)
     if logger:
         logger.log(f"  - 😋 Loss history saved in {out_path.name}")
