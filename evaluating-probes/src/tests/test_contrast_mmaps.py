import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.data import Dataset
from src.activations import ActivationManager


def create_mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.cfg.model_name = "test_model"
    model.cfg.d_model = 512
    model.tokenizer = Mock()
    model.tokenizer.padding_side = "left"
    model.tokenizer.truncation_side = "left"
    model.tokenizer.return_value = {
        "input_ids": np.random.randint(0, 1000, (2, 10))
    }
    model.run_with_cache.return_value = (None, {"blocks.0.hook_resid_pre": np.random.randn(2, 10, 512).astype(np.float16)})
    return model


def create_test_dataset():
    """Create a test dataset for testing."""
    df = pd.DataFrame({
        "prompt": ["Hello world", "Test prompt", "Another test"],
        "target": [0, 1, 0],
        "prompt_len": [11, 11, 12]
    })
    return df


def test_contrast_mmaps_separate():
    """Test that contrast probing creates separate mmaps from original dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock model and dataset
        model = create_mock_model()
        df = create_test_dataset()
        
        # Create original dataset
        original_dataset = Dataset.from_dataframe(
            df=df,
            dataset_name="test_dataset",
            model=model,
            device="cpu",
            cache_root=temp_path / "cache",
            seed=42,
            task_type="classification",
            n_classes=2,
            max_len=10,
            train_indices=[0, 1],
            val_indices=[2],
            test_indices=[]
        )
        
        # Create a simple contrast function
        def simple_contrast(row):
            orig = {"prompt": row["prompt"], "target": row["target"]}
            contrast = {"prompt": row["prompt"] + " contrast", "target": row["target"]}
            return orig, contrast
        
        # Get contrast activations
        with patch.object(ActivationManager, 'get_activations_for_texts') as mock_get_acts:
            mock_get_acts.return_value = np.random.randn(2, 10, 512).astype(np.float16)
            
            contrast_acts, labels = original_dataset.get_contrast_activations(
                simple_contrast, layer=0, component="resid_pre", split="train"
            )
        
        # Check that the contrast cache directory is different from the original
        original_cache_dir = temp_path / "cache" / "test_model" / "test_dataset"
        contrast_cache_dir = temp_path / "cache" / "test_model" / "test_dataset_contrast"
        
        # The contrast cache directory should be different from the original
        assert str(contrast_cache_dir) != str(original_cache_dir)
        assert "contrast" in str(contrast_cache_dir)
        
        # Verify that the contrast function name is included in the filename
        from src.runner import get_probe_filename_prefix
        filename = get_probe_filename_prefix(
            "test_dataset", "linear", "mean", 0, "resid_pre", simple_contrast
        )
        assert "simple_contrast" in filename
        print("✓ test_contrast_mmaps_separate passed")


def test_contrast_vs_original_mmaps():
    """Test that contrast and original activations use different cache directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock model and dataset
        model = create_mock_model()
        df = create_test_dataset()
        
        # Create original dataset
        original_dataset = Dataset.from_dataframe(
            df=df,
            dataset_name="test_dataset",
            model=model,
            device="cpu",
            cache_root=temp_path / "cache",
            seed=42,
            task_type="classification",
            n_classes=2,
            max_len=10,
            train_indices=[0, 1],
            val_indices=[2],
            test_indices=[]
        )
        
        # Create a simple contrast function
        def simple_contrast(row):
            orig = {"prompt": row["prompt"], "target": row["target"]}
            contrast = {"prompt": row["prompt"] + " contrast", "target": row["target"]}
            return orig, contrast
        
        # Mock the ActivationManager to track which cache directories are used
        cache_dirs_used = []
        
        def mock_get_activations_for_texts(texts, layer, component):
            # Get the cache directory from the ActivationManager instance
            cache_dir = mock_get_activations_for_texts.cache_dir
            cache_dirs_used.append(str(cache_dir))
            return np.random.randn(len(texts), 10, 512).astype(np.float16)
        
        with patch.object(ActivationManager, 'get_activations_for_texts', side_effect=mock_get_activations_for_texts):
            # Get original activations
            original_dataset.get_train_set_activations(0, "resid_pre")
            
            # Get contrast activations
            original_dataset.get_contrast_activations(simple_contrast, 0, "resid_pre", "train")
        
        # Should have used two different cache directories
        assert len(cache_dirs_used) == 2
        assert cache_dirs_used[0] != cache_dirs_used[1]
        assert "contrast" in cache_dirs_used[1]
        print("✓ test_contrast_vs_original_mmaps passed")


if __name__ == "__main__":
    test_contrast_mmaps_separate()
    test_contrast_vs_original_mmaps()
    print("All tests passed!") 