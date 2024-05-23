import os

# Get the project directory / from /src/utils/constants.py
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default directories, can be configured by configure_paths in setup.py
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints/")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets/")
LOG_DIR = os.path.join(PROJECT_DIR, "logs/")
SRC_DIR = os.path.join(PROJECT_DIR, "src/")

# Dataset paths
MEDMNIST_DATA_DIR = os.path.join(DATASETS_DIR, "medmnist/")
MIMETA_DATA_DIR = os.path.join(DATASETS_DIR, "mimeta/")

# Path to model checkpoints
SIMCLR_CHECKPOINT_PATH = os.path.join(CKPT_DIR, "simclr")
DINO_CHECKPOINT_PATH = os.path.join(CKPT_DIR, "dino")
DOWNSTREAM_CHECKPOINT_PATH = os.path.join(CKPT_DIR, "eval")

# Path to logs (wandb/tb)
SIMCLR_LOG_PATH = os.path.join(LOG_DIR, "simclr")
DOWNSTREAM_LOG_PATH = os.path.join(LOG_DIR, "eval")
