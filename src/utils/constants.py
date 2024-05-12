import os

NUM_WORKERS = os.cpu_count()
OUTPUT_DIR = "output/"

MEDMNIST_DATA_DIR = "datasets/medmnist/"
MIMETA_DATA_DIR = "datasets/mimeta/"

#########
# layout
# - ssl
#   - simclr
#     - models
#       - model_name.ckpt
#     - tb_logs
#       - model_name_simclr
#   - byol
#     - models
#
#########


# path to models
SSL_PATH = "ssl/"
MODEL_DIR = "models/"
SIMCLR_CHECKPOINT_PATH = f"src/{SSL_PATH}simclr/{MODEL_DIR}"
BYOL_CHECKPOINT_PATH = f"src/{SSL_PATH}byol/{MODEL_DIR}"
DINO_CHECKPOINT_PATH = f"src/{SSL_PATH}dino/{MODEL_DIR}"

# path to tensorboard logs
SIMCLR_TB_PATH = f"{SIMCLR_CHECKPOINT_PATH}logs/"
BYOL_TB_PATH = f"{BYOL_CHECKPOINT_PATH}logs/"
DINO_TB_PATH = f"{DINO_CHECKPOINT_PATH}logs/"
