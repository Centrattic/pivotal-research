# configs.py
DATA_DIR = "../datasets/cleaned"
# MODEL_DIR = "models"
RESULTS_DIR = "../results"

# Default settings
DEFAULT_MODEL = "gpt2-medium"
SUPPORTED_MODELS = [
    "gpt2", "gpt2-medium", "gpt2-large",
]
DEFAULT_DEVICE = "cuda"  # or "cpu"