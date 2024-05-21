import os
from pathlib import Path

from omegaconf import OmegaConf
import torchvision.models as models

from src.utils.augmentations import AugmentationSequenceType
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
            print(f"Number of workers is set to the number of CPU cores ({os.cpu_count()}) since it was set to -1")
        
        assert self.config.Device.num_workers > 0
        if self.config.Device.device == "gpu":
            assert self.config.Device.gpu_id >= 0
        
        assert self.config.Dataset.name in DatasetEnum.__members__, \
            f"Invalid dataset name: {self.config.Dataset.dataset_name} (one of {DatasetEnum.__members__})"
        
        if self.config.Dataset.name == "medmnist":
            assert self.config.Dataset.params.medmnist_flag in MedMNISTCategory.__members__, \
                f"Invalid data flag for MedMNIST: {self.config.Dataset.data_flag} (one of {MedMNISTCategory.__members__})"
        
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
            assert _train_cfg.Pretrain.ssl_method in SSLMethod.__members__, \
                f"Invalid SSL method: {_train_cfg.Pretrain.ssl_method} (one of {SSLMethod.__members__})"

            if "augmentations" in _train_cfg.Pretrain:
                if isinstance(_train_cfg.Pretrain.augmentations, list):
                    raise NotImplementedError("Custom augmentation sequences are not supported yet.")
                    # TODO: Assert if the list contains valid torchvision transforms
                else:
                    assert _train_cfg.Pretrain.augmentations in AugmentationSequenceType.__members__ or _train_cfg.Pretrain.augmentations in [None, "None"], \
                        f"Invalid augmentation sequence: {_train_cfg.Pretrain.augmentations} (one of {AugmentationSequenceType.__members__})"

            model_names = models.list_models()
            assert _train_cfg.Pretrain.params.encoder in model_names, \
                f"Invalid model name: {_train_cfg.Pretrain.params.encoder} (one of {model_names})"
            
            assert _train_cfg.Pretrain.params.hidden_dim > 0
            assert _train_cfg.Pretrain.params.output_dim > 0

            assert _train_cfg.Pretrain.params.temperature >= 0
            assert _train_cfg.Pretrain.params.n_views > 0
        else:
            assert "Downstream" in _train_cfg, "Either Pretrain or Downstream field must be provided."

            assert _train_cfg.Downstream.method in DownstreamMethod.__members__, \
                f"Invalid downstream method: {_train_cfg.Downstream.method} (one of {DownstreamMethod.__members__})"
            
            if "augmentations" in _train_cfg.Downstream:
                if isinstance(_train_cfg.Downstream.augmentations, list):
                    raise NotImplementedError("Custom augmentation sequences are not supported yet.")
                    # TODO: Assert if the list contains valid torchvision transforms
                else:
                    assert _train_cfg.Downstream.augmentations in AugmentationSequenceType.__members__ or _train_cfg.Downstream.augmentations in [None, "None"], \
                        f"Invalid augmentation sequence: {_train_cfg.Downstream.augmentations} (one of {AugmentationSequenceType.__members__})"

            model_names = models.list_models()
            assert _train_cfg.Downstream.params.encoder in model_names, \
                f"Invalid model name: {_train_cfg.Downstream.params.encoder} (one of {model_names})"

            assert _train_cfg.Downstream.params.hidden_dim > 0
        
        assert self.config.Logging.tool in LoggingTools.__members__, \
            f"Invalid logging tool: {self.config.Logging.tool} (one of {LoggingTools.__members__})"
        
        assert self.config.Logging.log_steps > 0


    # TODO: Verify whether the casting is necessary or not. Most likely int and float are already casted, might remove the parse function
    def _cast_cfg(self):
        "Cast the config values to their respective intended types"
        self.config.Device.num_workers = self._parse_cfg_str(self.config.Device.num_workers, int)

        self.config.Dataset.name = DatasetEnum[self.config.Dataset.name]
        self.config.Dataset.params.medmnist_flag = MedMNISTCategory[self.config.Dataset.params.medmnist_flag]
        self.config.Dataset.params.image_size = self._parse_cfg_str(self.config.Dataset.params.image_size, int)
        
        _train_cfg = self.config.Training
        _train_cfg.params.batch_size = self._parse_cfg_str(_train_cfg.params.batch_size, int)
        _train_cfg.params.max_epochs = self._parse_cfg_str(_train_cfg.params.max_epochs, int)
        _train_cfg.params.learning_rate = self._parse_cfg_str(_train_cfg.params.learning_rate, float)
        _train_cfg.params.weight_decay = self._parse_cfg_str(_train_cfg.params.weight_decay, float)

        if "Pretrain" in _train_cfg:
            _train_cfg.Pretrain.ssl_method = SSLMethod[_train_cfg.Pretrain.ssl_method]
            _train_cfg.Pretrain.params.hidden_dim = self._parse_cfg_str(_train_cfg.Pretrain.params.hidden_dim, int)
            _train_cfg.Pretrain.params.output_dim = self._parse_cfg_str(_train_cfg.Pretrain.params.output_dim, int)
            _train_cfg.Pretrain.params.temperature = self._parse_cfg_str(_train_cfg.Pretrain.params.temperature, float)
            _train_cfg.Pretrain.params.n_views = self._parse_cfg_str(_train_cfg.Pretrain.params.n_views, int)

            # If augmentation is not specified, set the default augmentation sequence to "default" for self-supervised learning.
            if "augmentations" not in _train_cfg.Pretrain or _train_cfg.Pretrain.augmentations in [None, "None"]:
                _train_cfg.Pretrain.augmentations = AugmentationSequenceType.DEFAULT
            elif isinstance(_train_cfg.Pretrain.augmentations, str):
                _train_cfg.Pretrain.augmentations = AugmentationSequenceType[_train_cfg.Pretrain.augmentations]
            else:  # List[torchvision.transforms]
                raise NotImplementedError("Custom augmentation sequences are not supported yet.")

        if "Downstream" in _train_cfg:
            _train_cfg.Downstream.method = DownstreamMethod[_train_cfg.Downstream.method]
            _train_cfg.Downstream.params.ssl_method = SSLMethod[_train_cfg.Downstream.params.ssl_method]
            _train_cfg.Downstream.params.hidden_dim = self._parse_cfg_str(_train_cfg.Downstream.params.hidden_dim, int)

            # If augmentation is not specified, set the default augmentation sequence to "preprocess" for downstream tasks.
            if "augmentations" not in self.config.Training.Downstream:
                self.config.Training.Downstream.augmentations = AugmentationSequenceType.PREPROCESS
            elif isinstance(self.config.Training.Downstream.augmentations, str):
                self.config.Training.Downstream.augmentations = AugmentationSequenceType[self.config.Training.Downstream.augmentations]
            else:  # List[torchvision.transforms]
                raise NotImplementedError("Custom augmentation sequences are not supported yet.")

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

        # Get the default values for the config fields that are not provided
        # Note: in OmegaConf.merge, the second argument has higher priority

        self.config.Device = OmegaConf.merge(self._defaults.Device, self.config.Device)
        self.config.Dataset = OmegaConf.merge(self._defaults.Dataset, self.config.Dataset)
        self.config.Training = OmegaConf.merge(self._defaults.Training.params, self.config.Training.params)

        # Does not automatically merge Pretrain and Downstream fields
        if "Pretrain" in self.config.Training:
            self.config.Training.Pretrain = OmegaConf.merge(self._defaults.Training.Pretrain, self.config.Training.Pretrain)
        elif "Downstream" in self.config.Training:
            self.config.Training.Downstream = OmegaConf.merge(self._defaults.Training.Downstream, self.config.Training.Downstream)

        self.config.Logging = OmegaConf.merge(self._defaults.Logging, self.config.Logging)
