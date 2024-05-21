import os
from pathlib import Path

from omegaconf import OmegaConf
import torchvision.models as models

import src.utils.constants as const
from src.utils.enums import DatasetEnum, MedMNISTCategory, SSLMethod, \
    DownstreamMethod, LoggingTools


class Config:
    def _parse_cfg_str(self, inp, casttype):
        if casttype is None:
            return None if inp == "None" else inp
        else:
            return casttype(inp) if isinstance(inp, str) else inp

    def _sanitize_cfg(self):
        if self.config.Device.num_workers == -1:
            self.config.Device.num_workers = os.cpu_count()
            # logger.info("Number of workers is set to the number of CPU cores (%d) since it was set to -1", os.cpu_count())
        
        assert self.config.Device.num_workers > 0
        if self.config.Device.device == "gpu":
            assert self.config.Device.gpu_id >= 0
        
        assert self.config.Dataset.name in DatasetEnum, \
            f"Invalid dataset name: {self.config.Dataset.dataset_name} (one of {DatasetEnum})"
        
        if self.config.Dataset.name == "medmnist":
            assert self.config.Dataset.params.medmnist_flag in MedMNISTCategory, \
                f"Invalid data flag for MedMNIST: {self.config.Dataset.data_flag} (one of {MedMNISTCategory})"
        
        assert self.config.Dataset.params.image_size in [28, 64, 128, 224], \
            f"Invalid image size: {self.config.Dataset.image_size} (valid sizes: 28, 64, 128, 224)"
    
        _train_cfg = self.config.Training
        assert _train_cfg.params.batch_size > 0
        assert _train_cfg.params.max_epochs > 0
        assert _train_cfg.params.learning_rate > 0
        assert _train_cfg.params.weight_decay >= 0

        # Raise error if both Pretrain and Downstream fields are provided
        if "Pretrain" in _train_cfg and "Downstream" in _train_cfg:
            raise ValueError("Both Pretrain and Downstream fields are provided under Training. Please provide only one of them.")

        if "Pretrain" in _train_cfg:
            assert _train_cfg.Pretrain.ssl_method in SSLMethod, \
                f"Invalid SSL method: {_train_cfg.Pretrain.ssl_method} (one of {SSLMethod})"

            model_names = models.list_models()
            assert _train_cfg.Pretrain.params in model_names, \
                f"Invalid model name: {_train_cfg.Pretrainparams} (one of {model_names})"
            
            assert _train_cfg.Pretrain.params.hidden_dim > 0
            assert _train_cfg.Pretrain.params.feature_dim > 0
            assert _train_cfg.Pretrain.params.output_dim > 0

            assert _train_cfg.Pretrain.params.temperature >= 0
            assert _train_cfg.Pretrain.params.n_views > 0
        else:
            assert "Downstream" in _train_cfg, "Either Pretrain or Downstream field must be provided."

            assert _train_cfg.Downstream.method in DownstreamMethod, \
                f"Invalid downstream method: {_train_cfg.Downstream.method} (one of {DownstreamMethod})"
            
            model_names = models.list_models()
            assert _train_cfg.Downstream.params in model_names, \
                f"Invalid model name: {_train_cfg.Downstream.params} (one of {model_names})"

            assert _train_cfg.Downstream.params.hidden_dim > 0

            assert _train_cfg.Downstream.params.batch_size > 0
            assert _train_cfg.Downstream.params.learning_rate > 0
            assert _train_cfg.Downstream.params.max_epochs > 0
            assert _train_cfg.Downstream.params.weight_decay >= 0
        
        assert self.config.Logging.tool in LoggingTools, \
            f"Invalid logging tool: {self.config.Logging.tool} (one of {LoggingTools})"
        
        assert self.config.Logging.log_steps > 0


    # TODO: Verify whether the casting is necessary or not
    def _cast_cfg(self):
        "Cast the config values to their respective intended types"
        self.config.Device.num_workers = self._parse_cfg_str(self.config.Device.num_workers, int)

        self.config.Dataset.params.medmnist_flag = MedMNISTCategory[self.config.Dataset.params.medmnist_flag]
        self.config.Dataset.params.image_size = self._parse_cfg_str(self.config.Dataset.params.image_size, int)
        
        _train_cfg = self.config.Training
        _train_cfg.batch_size = self._parse_cfg_str(_train_cfg.batch_size, int)
        _train_cfg.max_epochs = self._parse_cfg_str(_train_cfg.max_epochs, int)
        _train_cfg.learning_rate = self._parse_cfg_str(_train_cfg.learning_rate, float)
        _train_cfg.weight_decay = self._parse_cfg_str(_train_cfg.weight_decay, float)

        if "Pretrain" in _train_cfg:
            _train_cfg.Pretrain.ssl_method = SSLMethod[_train_cfg.Pretrain.ssl_method]
            _train_cfg.Pretrain.params.hidden_dim = self._parse_cfg_str(_train_cfg.Pretrain.params.hidden_dim, int)
            _train_cfg.Pretrain.params.feature_dim = self._parse_cfg_str(_train_cfg.Pretrain.params.feature_dim, int)
            _train_cfg.Pretrain.params.output_dim = self._parse_cfg_str(_train_cfg.Pretrain.params.output_dim, int)
            _train_cfg.Pretrain.params.temperature = self._parse_cfg_str(_train_cfg.Pretrain.params.temperature, float)
            _train_cfg.Pretrain.params.n_views = self._parse_cfg_str(_train_cfg.Pretrain.params.n_views, int)
        if "Downstream" in _train_cfg:
            _train_cfg.Downstream.method = DownstreamMethod[_train_cfg.Downstream.method]
            _train_cfg.Downstream.params.hidden_dim = self._parse_cfg_str(_train_cfg.Downstream.params.hidden_dim, int)

        self.config.Logging.tool = LoggingTools[self.config.Logging.tool]
        self.config.Logging.log_steps = self._parse_cfg_str(self.config.Logging.log_steps, int)


    def __init__(self, config_path, defaults_path=(Path(__file__).parent / "default_config.yaml").resolve()):
        """
        Load the config file and merge it with the default config file
        :param config_path: Path to the config YAML file
        :param defaults_path: Path to the default config YAML file. Fields that are not provided in the config file will be taken from here.
        """

        self.config = OmegaConf.load(config_path)
        self._defaults = OmegaConf.load(defaults_path)

        self._sanitize_cfg()
        self._cast_cfg()
        
        # TODO: Merge the config with the default config file, except:
        # Do not create Pretrain or Downstream fields if they don't exist. If both doesn't exist, create Pretrain.
        # If Pretrain/Downstream does not have augmentations, give empty list.
        ...
