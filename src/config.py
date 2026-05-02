from pathlib import Path

# Randomness
RANDOM_SEED = 42

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Dataset
DATASET_SLUG = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
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
GROUPED_SPLITS_DIR = DATA_DIR / "splits_grouped"

MODELS_DIR = PROJECT_ROOT / "models"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CM_DIR = OUTPUTS_DIR / "confusion_matrices"
LOGS_DIR = OUTPUTS_DIR / "logs"
XAI_DIR = OUTPUTS_DIR / "xai_examples"

# Task 1 quick-reorder recommender
TASK1_DATA_DIR = DATA_DIR / "task1" / "desd_seed_export"
TASK1_OUTPUT_DIR = OUTPUTS_DIR / "task1_recommender"
