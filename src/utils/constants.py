import os

# Accepted values for config fields
LOGGGING_TOOLS = ["wandb", "tb", "none"]
SSL_METHODS = ["simclr", "dino"]
EVAL_METHODS = ["linear", "mlp"]

# Constants for directories
OUTPUT_DIR = "output/"
DATASETS_DIR = "datasets/"
SRC_DIR = "src/"
SSL_DIR = "ssl/"
MODEL_DIR = "ckpts/"
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
DOWNSTREAM_LOG_PATH = os.path.join(SRC_DIR, DOWNSTREAM_DIR, "eval")

# downstream
DOWNSTREAM_CHECKPOINT_PATH = os.path.join(SRC_DIR, DOWNSTREAM_DIR, "eval", MODEL_DIR)
