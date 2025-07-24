
import numpy as np

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
