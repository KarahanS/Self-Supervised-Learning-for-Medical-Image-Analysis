import torch
import numpy as np
import random
import os


def setup_device(seed):
    """
    Set up GPU device if available, otherwise set up CPU.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        set_seed(seed)


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
    print(f"Random seed set as {seed}")


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
