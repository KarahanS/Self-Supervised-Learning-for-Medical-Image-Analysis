import os
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, ConfigKeyError
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
        if self.config.Device.use_gpu:
            assert self.config.Device.gpu_id >= -1
        
        assert self.config.Dataset.name in DatasetEnum.__members__, \
            f"Invalid dataset name: {self.config.Dataset.name} (one of {list(DatasetEnum.__members__.keys())})"
        
        if self.config.Dataset.name == "medmnist":
            assert self.config.Dataset.params.medmnist_flag in MedMNISTCategory.__members__, \
                f"Invalid data flag for MedMNIST: {self.config.Dataset.data_flag} (one of {list(MedMNISTCategory.__members__.keys())})"
        
        if not self.config.Dataset.path.endswith("/"):
            self.config.Dataset.path += "/"

        assert self.config.Dataset.params.image_size in [28, 64, 128, 224], \
            f"Invalid image size: {self.config.Dataset.image_size} (valid sizes: 28, 64, 128, 224)"
    
        _train_cfg = self.config.Training
        assert _train_cfg.params.batch_size > 0
        assert _train_cfg.params.max_epochs > 0
        assert _train_cfg.params.learning_rate > 0
        assert _train_cfg.params.weight_decay >= 0

        if not _train_cfg.checkpoints.path.endswith("/"):
            _train_cfg.checkpoints.path += "/"
        assert _train_cfg.checkpoints.save_steps >= 0

        # Raise error if both Pretrain and Downstream fields are provided
        if "Pretrain" in _train_cfg and "Downstream" in _train_cfg:
            raise ValueError("Both Pretrain and Downstream fields are provided under Training. Please provide only one of them.")

        if "Pretrain" in _train_cfg:
            assert _train_cfg.Pretrain.ssl_method in SSLMethod.__members__, \
                f"Invalid SSL method: {_train_cfg.Pretrain.ssl_method} (one of {list(SSLMethod.__members__.keys())})"

            if isinstance(_train_cfg.Pretrain.augmentations, list):
                raise NotImplementedError("Custom augmentation sequences are not supported yet.")
                # TODO: Assert if the list contains valid torchvision transforms
            else:
                assert _train_cfg.Pretrain.augmentations in AugmentationSequenceType.__members__ or _train_cfg.Pretrain.augmentations in [None, "None"], \
                    f"Invalid augmentation sequence: {_train_cfg.Pretrain.augmentations} (one of {list(AugmentationSequenceType.__members__.keys())})"

            model_names = models.list_models()
            assert _train_cfg.Pretrain.params.encoder in model_names, \
                f"Invalid model name: {_train_cfg.Pretrain.params.encoder} (one of {model_names})"
            
            assert _train_cfg.Pretrain.params.hidden_dim > 0
            assert _train_cfg.Pretrain.params.output_dim > 0

            assert _train_cfg.Pretrain.params.temperature >= 0
            assert _train_cfg.Pretrain.params.n_views > 0
        else:
            assert "Downstream" in _train_cfg, "Either Pretrain or Downstream field must be provided."

            assert _train_cfg.Downstream.eval_method in DownstreamMethod.__members__, \
                f"Invalid downstream method: {_train_cfg.Downstream.eval_method} (one of {list(DownstreamMethod.__members__.keys())})"
            
            assert _train_cfg.Downstream.ssl_method in SSLMethod.__members__, \
                f"Invalid downstream method: {_train_cfg.Downstream.ssl_method} (one of {list(SSLMethod.__members__.keys())})"
            
            if isinstance(_train_cfg.Downstream.augmentations, list):
                raise NotImplementedError("Custom augmentation sequences are not supported yet.")
                # TODO: Assert if the list contains valid torchvision transforms
            else:
                assert _train_cfg.Downstream.augmentations in AugmentationSequenceType.__members__ or _train_cfg.Downstream.augmentations in [None, "None"], \
                    f"Invalid augmentation sequence: {_train_cfg.Downstream.augmentations} (one of {list(AugmentationSequenceType.__members__.keys())})"

            model_names = models.list_models()
            assert _train_cfg.Downstream.params.encoder in model_names, \
                f"Invalid model name: {_train_cfg.Downstream.params.encoder} (one of {model_names})"

            assert _train_cfg.Downstream.params.hidden_dim > 0
        
        if not self.config.Logging.path.endswith("/"):
            self.config.Logging.path += "/"
        assert self.config.Logging.tool in LoggingTools.__members__, \
            f"Invalid logging tool: {self.config.Logging.tool} (one of {list(LoggingTools.__members__.keys())})"
        
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

            if isinstance(_train_cfg.Pretrain.augmentations, str):
                _train_cfg.Pretrain.augmentations = AugmentationSequenceType[_train_cfg.Pretrain.augmentations]
            else:  # List[torchvision.transforms]
                raise NotImplementedError("Custom augmentation sequences are not supported yet.")

        if "Downstream" in _train_cfg:
            _train_cfg.Downstream.eval_method = DownstreamMethod[_train_cfg.Downstream.eval_method]
            _train_cfg.Downstream.ssl_method = SSLMethod[_train_cfg.Downstream.ssl_method]
            _train_cfg.Downstream.params.hidden_dim = self._parse_cfg_str(_train_cfg.Downstream.params.hidden_dim, int)
            
            if isinstance(self.config.Training.Downstream.augmentations, str):
                self.config.Training.Downstream.augmentations = AugmentationSequenceType[self.config.Training.Downstream.augmentations]
            else:  # List[torchvision.transforms]
                raise NotImplementedError("Custom augmentation sequences are not supported yet.")

        self.config.Logging.tool = LoggingTools[self.config.Logging.tool]
        self.config.Logging.log_steps = self._parse_cfg_str(self.config.Logging.log_steps, int)

    def __init__(self, config_path = None, config = None, defaults_path=(Path(__file__).parent / "defaults.yaml").resolve()):
        """
        Load the config file and merge it with the default config file
        :param config_path: Path to the config YAML file
        :param defaults_path: Path to the default config YAML file. Fields that are not provided in the config file will be taken from here.
        """

        if config_path:
            self.config = OmegaConf.load(config_path)
            self._defaults = OmegaConf.load(defaults_path)

            # Get the default values for the config fields that are not provided
            # Note: in OmegaConf.merge, the second argument has higher priority

            for key in ["Device", "Dataset", "Logging"]:
                if key in self.config:
                    self.config[key] = OmegaConf.merge(self._defaults[key], self.config[key])
                else:
                    self.config[key] = self._defaults[key]
            
            for key in ["params", "checkpoints"]:
                if key in self.config.Training:
                    self.config.Training[key] = OmegaConf.merge(self._defaults.Training[key], self.config.Training[key])
                else:
                    self.config.Training[key] = self._defaults.Training[key]
            
            # Do not automatically merge Pretrain and Downstream fields
            if "Pretrain" in self.config.Training:
                self.config.Training.Pretrain = OmegaConf.merge(self._defaults.Training.Pretrain, self.config.Training.Pretrain)
            elif "Downstream" in self.config.Training:
                self.config.Training.Downstream = OmegaConf.merge(self._defaults.Training.Downstream, self.config.Training.Downstream)
            self._sanitize_cfg()
            self._cast_cfg()
        else:
            self.config = config
       

    def dump_config(self, path):
        OmegaConf.save(self.config, path)

    def convert_to_dict(self):
        return OmegaConf.to_container(self.config)
    
    @classmethod
    def convert_from_dict(cls, cfg_dict):
        config = OmegaConf.create(cfg_dict)
        return cls(config=config)


    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except ConfigAttributeError:
            # For safety - should not be necessary since all fields are already merged
            # Also captures errors arising from subsequent accesses when accessed through root Config object
            retval = getattr(self._defaults, item)

            return retval

    def __getitem__(self, item):
        try:
            return self.config[item]
        except ConfigKeyError:
            # For safety - should not be necessary since all fields are already merged
            # Also captures errors arising from subsequent accesses when accessed through root Config object
            retval = self._defaults[item]

            return retval

    def __iter__(self):
        return iter(self.config)
