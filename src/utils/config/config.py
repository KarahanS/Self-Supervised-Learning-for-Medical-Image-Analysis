import os
from pathlib import Path

from omegaconf import OmegaConf

import src.utils.constants as const
from src.utils.enums import DatasetEnum, MedMNISTCategory, AugmentationSequenceType


class Config:
    def __init__(self, config_path, defaults_path=(Path(__file__).parent / "default_config.yaml").resolve()):
        """
        Load the config file and merge it with the default config file
        :param config_path: Path to the config YAML file
        :param defaults_path: Path to the default config YAML file. Fields that are not provided in the config file will be taken from here.
        """

        self.config = OmegaConf.load(config_path)
        self._defaults = OmegaConf.load(defaults_path)

        # Sanitize the config file

        assert self.config.Dataset.name in DatasetEnum, \
            f"Invalid dataset name: {self.config.Dataset.dataset_name} (one of {DatasetEnum})"
        
        if self.config.Dataset.name == "medmnist":
            assert self.config.Dataset.params.medmnist_flag in MedMNISTCategory, \
                f"Invalid data flag for MedMNIST: {self.config.Dataset.data_flag} (one of {MedMNISTCategory})"
        
        assert self.config.Dataset.params.image_size in [28, 64, 128, 224], \
            f"Invalid image size: {self.config.Dataset.image_size} (valid sizes: 28, 64, 128, 224)"
    
        # Raise error if both Pretrain and Downstream fields are provided
        if "Pretrain" in self.config and "Downstream" in self.config:
            raise ValueError("Both Pretrain and Downstream fields are provided. Please provide only one of them.")

        assert self.config.SSL.method in const.SSL_METHODS, \
            f"Invalid SSL method: {self.config.SSL.method} (one of {const.SSL_METHODS})"

        assert self.config.Downstream.method in const.EVAL_METHODS, \
            f"Invalid downstream method: {self.config.Downstream.method} (one of {const.EVAL_METHODS})"
        
        assert self.config.Logging.tool in const.LOGGGING_TOOLS, \
            f"Invalid logging tool: {self.config.Logging.tool} (one of {const.LOGGGING_TOOLS})"


        # TODO: Create dir.s if they don't exist

        # TODO: Merge the config with the default config file, except:
        # Do not create Pretrain or Downstream fields if they don't exist. If both doesn't exist, create Pretrain.
        # If Pretrain/Downstream does not have augmentations, give empty list.
        ...
