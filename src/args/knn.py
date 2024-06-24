import omegaconf
from omegaconf import OmegaConf
from src.ssl.methods.base import BaseMethod
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.misc import omegaconf_select
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from src.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True

_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "stl10",
    "imagenet",
    "imagenet100",
    "custom",  # custom for pretraining
    "bloodmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "breastmnist",
    "organamnist",
    "organcmnist",
    "pneumoniamnist",
    "retinamnist",
    "pathmnist",
    "tissuemnist",
    "organsmnist",
]

def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for dataset config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "data.dataset")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)

    return cfg

def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for wandb config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)

    return cfg

def add_and_assert_lightning_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for Pytorch Lightning config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.seed = omegaconf_select(cfg, "seed", 5)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg


def parse_cfg(cfg: omegaconf.DictConfig):

    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for dali
    if _dali_available:
        cfg = ClassificationDALIDataModule.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    #cfg = add_and_assert_lightning_cfg(cfg)

    # backbone
    assert not omegaconf.OmegaConf.is_missing(cfg, "backbone.name")
    assert cfg.backbone.name in BaseMethod._BACKBONES

    # backbone kwargs
    cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

    assert not omegaconf.OmegaConf.is_missing(cfg, "pretrained_feature_extractor")

    cfg.pretrain_method = omegaconf_select(cfg, "pretrain_method", None)
    
    # augmentation related (crop size and custom mean/std values for normalization)
    cfg.data.augmentations = omegaconf_select(cfg, "data.augmentations", {})
    cfg.data.augmentations.crop_size = omegaconf_select(
        cfg, "data.augmentations.crop_size", 64
    )
    cfg.data.augmentations.mean = omegaconf_select(
        cfg, "data.augmentations.mean", IMAGENET_DEFAULT_MEAN
    )
    cfg.data.augmentations.std = omegaconf_select(
        cfg, "data.augmentations.std", IMAGENET_DEFAULT_STD
    )

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # add knn args
    # cfg.knn.k = omegaconf_select(cfg, "k", [200])
    # cfg.knn.T = omegaconf_select(cfg, "T", [0.07])
    # cfg.knn.distance_fx = omegaconf_select(cfg, "distance_function", ["cosine"])
    # cfg.knn.temperature = omegaconf_select(cfg, "temperature", [0.07])
    # cfg.knn.feature_type = omegaconf_select(cfg, "feature_type", ["backbone"])
    # cfg.backbone = omegaconf_select(cfg, "backbone", "resnet50")
    # cfg.pretrain_method = omegaconf_select(cfg, "pretrain_method", None)
    # cfg.knn.batch_size = omegaconf_select(cfg, "batch_size", 256)

    return cfg
