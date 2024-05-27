import torch
import numpy as np
import random
import os
import datetime
import logging
import src.utils.constants as const
import sys
import time

class TimestampedFile:
    def __init__(self, file):
        self.file = file
        self.is_at_start_of_line = True

    def write(self, data):
        if self.is_at_start_of_line:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.file.write(f"{timestamp} ")
        self.file.write(data)
        self.is_at_start_of_line = data.endswith('\n')

    def flush(self):
        self.file.flush()


def setup_device(cfg):
    """
    Set up GPU device if available, otherwise set up CPU.
    """
    device_cfg = cfg.Device

    if device_cfg.use_gpu and torch.cuda.is_available():
        if device_cfg.gpu_id >= 0:
            device = torch.device("cuda:" + str(device_cfg.gpu_id))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.backends.cudnn.enabled = False  # Set to True for faster training but more memory usage
    logging.info(f"Using device: {device}")
    # logging.info()
    # logging.info(torch.cuda.current(device))

    set_seed(cfg.seed)  # TODO: Verify if it works with CPU


def get_device():
    """
    Get the device (GPU or CPU) based on availability.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# seed for reproducibility
def set_seed(seed: int = 7) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def get_accelerator_info():
    """
    Get accelerator type (GPU or CPU) and memory usage.
    """

    if torch.cuda.is_available():
        accelerator = "gpu"
        num_threads = torch.cuda.device_count()
    else:
        accelerator = "cpu"
        num_threads = torch.get_num_threads()

    # TODO: Is Multi-GPU supported? Otherwise set num_threads to 1
    return accelerator, num_threads


def configure_paths(cfg):
    """
    Configure paths for datasets, checkpoints, logs, and other outputs.
    """

    const.DATASETS_DIR = cfg.Dataset.path
    const.MEDMNIST_DATA_DIR = os.path.join(const.DATASETS_DIR, "medmnist/")
    const.MIMETA_DATA_DIR = os.path.join(const.DATASETS_DIR, "mimeta/")

    const.CKPT_DIR = cfg.Training.checkpoints.path
    const.SIMCLR_CHECKPOINT_PATH = os.path.join(const.CKPT_DIR, "simclr/")
    const.DINO_CHECKPOINT_PATH = os.path.join(const.CKPT_DIR, "dino/")
    const.DOWNSTREAM_CHECKPOINT_PATH = os.path.join(const.CKPT_DIR, "eval/")

    const.LOG_DIR = cfg.Logging.path
    const.SIMCLR_LOG_PATH = os.path.join(const.LOG_DIR, "simclr/")
    const.DOWNSTREAM_LOG_PATH = os.path.join(const.LOG_DIR, "eval/")

    # Create directories if they do not exist

    os.makedirs(const.DATASETS_DIR, exist_ok=True)
    os.makedirs(const.MEDMNIST_DATA_DIR, exist_ok=True)
    os.makedirs(const.MIMETA_DATA_DIR, exist_ok=True)

    os.makedirs(const.CKPT_DIR, exist_ok=True)
    os.makedirs(const.SIMCLR_CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(const.DINO_CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(const.DOWNSTREAM_CHECKPOINT_PATH, exist_ok=True)

    os.makedirs(const.LOG_DIR, exist_ok=True)
    os.makedirs(const.SIMCLR_LOG_PATH, exist_ok=True)
    os.makedirs(const.DOWNSTREAM_LOG_PATH, exist_ok=True)

# Create a logger
def setup_logger(cfg):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)  # Set lowest level to DEBUG

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Debug handler
    debug_handler = logging.FileHandler(os.path.join(const.RUN_LOG_PATH, f"{cfg.config_name}_{timestamp}_debug.log"))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    # Other handler
    other_handler = logging.FileHandler(os.path.join(const.RUN_LOG_PATH,f"{cfg.config_name}_{timestamp}.log"))
    other_handler.setLevel(logging.INFO)  # Set lowest level to INFO to exclude DEBUG logs
    other_handler.setFormatter(formatter)
    logger.addHandler(other_handler)

    # Stream handler
    logging.root.handlers = logger.handlers
    logging.root.setLevel(logger.level)


    logging.root.handlers = logger.handlers
    logging.root.setLevel(logger.level)

    return logger