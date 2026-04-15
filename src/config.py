from pathlib import Path

# Randomness
RANDOM_SEED = 42

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Dataset
NUM_CLASSES = 28
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Local project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CM_DIR = OUTPUTS_DIR / "confusion_matrices"
LOGS_DIR = OUTPUTS_DIR / "logs"
XAI_DIR = OUTPUTS_DIR / "xai_examples"