import os

# Constants for directories
OUTPUT_DIR = "output/"
DATASETS_DIR = "datasets/"
SRC_DIR = "src/"
SSL_DIR = "ssl/"
MODEL_DIR = "models/"
LOG_DIR = "logs/"
DOWNSTREAM_DIR = "downstream/"

# Derived constants
NUM_WORKERS = os.cpu_count()

# Dataset paths
MEDMNIST_DATA_DIR = os.path.join(DATASETS_DIR, "medmnist/")
MIMETA_DATA_DIR = os.path.join(DATASETS_DIR, "mimeta/")

# Path to model checkpoints
SIMCLR_CHECKPOINT_PATH = os.path.join(SRC_DIR, SSL_DIR, "simclr", MODEL_DIR)
DINO_CHECKPOINT_PATH = os.path.join(SRC_DIR, SSL_DIR, "dino", MODEL_DIR)

# Path to logs (wandb/tb)
SIMCLR_LOG_PATH = os.path.join(SRC_DIR, SSL_DIR, "simclr")

# Path to logs (wandb/tb) for downstream
LOGISTIC_REGRESSION_LOG_PATH = os.path.join(SRC_DIR, DOWNSTREAM_DIR, "linear_eval")

# downstream
LOGISTIC_REGRESSION_CHECKPOINT_PATH = os.path.join(
    SRC_DIR, DOWNSTREAM_DIR, "linear_eval"
)
