import torch
from pathlib import Path

HF_MODEL_NAME = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MAX_LENGTH = 128  # context length to project attentions into bins
DEFAULT_BATCH_SIZE = 16

THIS_DIR = Path(__file__).parent
CONFIG_PATH = THIS_DIR / "datasets.yml"
