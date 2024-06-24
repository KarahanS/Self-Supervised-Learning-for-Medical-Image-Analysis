import json
import os
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
from src.ssl.methods.base import BaseMethod
from src.utils.setup import get_device
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.data.classification_dataloader import prepare_data as prepare_data_classification
from src.data.loader.medmnist_loader import MedMNISTLoader, SplitType
from src.args.knn import parse_cfg
import hydra
from src.utils.eval import get_representations
from src.ssl.methods import METHODS
from src.utils.knn import WeightedKNNClassifier


@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    knn.update(train_features=train_features, train_targets=train_targets)
    knn.update(test_features=test_features, test_targets=test_targets)

    acc1, acc5, confusion_matrix, recall, precision = knn.compute()

    del knn

    return acc1, acc5, confusion_matrix, recall, precision


def build_data_loaders(dataset, image_size, batch_size, num_workers, root):

    loader = MedMNISTLoader(
        data_flag=dataset,
        download=True,
        batch_size=batch_size,
        size=image_size,
        num_workers=num_workers,
        root=root,
    )

    train_dataclass = loader.get_data(SplitType.TRAIN, root=root)
    val_dataclass = loader.get_data(SplitType.VAL, root=root)
    test_dataclass = loader.get_data(
        SplitType.TEST,
        root=root,
    )  # to be used afterwards for testing

    return loader, train_dataclass, val_dataclass, test_dataclass

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    if "vit" not in cfg.backbone.name:
        cfg.backbone.kwargs.pop('img_size',None)
        cfg.backbone.kwargs.pop('pretrained',None)

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    assert (
        ckpt_path.endswith(".ckpt")
        or ckpt_path.endswith(".pth")
        or ckpt_path.endswith(".pt")
    )

    loader, train_dataclass, val_dataclass, test_dataclass = build_data_loaders(
        cfg.data.dataset,
        image_size=cfg.data.image_size,
        batch_size=cfg.knn.batch_size,
        num_workers=cfg.data.num_workers,
        root=cfg.data.root,
    )

    device = get_device()

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)

    if cfg.backbone.name.startswith("resnet"):
        backbone.fc = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]

    backbone.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    train_feats_tuple = get_representations(backbone, train_dataclass, device)
    val_feats_tuple = get_representations(backbone, val_dataclass, device)
    test_feats_tuple = get_representations(backbone, test_dataclass, device)

    train_features = {"backbone": train_feats_tuple[0]}
    val_features = {"backbone": val_feats_tuple[0]}
    test_features = {"backbone": test_feats_tuple[0]}

    best_acc1 = 0
    best_result = None

    # print the parameters cfg.knn.k = omegaconf_select(cfg, "k", [200]) cfg.knn.T cfg.knn.distance_fx  cfg.knn.temperature  cfg.knn.feature_type  cfg.backbone  cfg.pretrain_method  cfg.knn.batch_size
    # print(f"Parameters: {cfg.knn.k}, {cfg.knn.distance_fx}, {cfg.knn.T}, {cfg.knn.feature_type}, {cfg.backbone}, {cfg.pretrain_method}, {cfg.knn.batch_size}")
    # print(cfg)

    # Calculate the total number of iterations
    if "cosine" and "euclidean" in cfg.knn.distance_fx:
        total_iterations = len(cfg.knn.feature_type) * len(cfg.knn.k) * (len(cfg.knn.T) + 1)
    elif "cosine" in cfg.knn.distance_fx:
        total_iterations = len(cfg.knn.feature_type) * len(cfg.knn.k) * (len(cfg.knn.T))
    else:
        total_iterations = len(cfg.knn.feature_type) * len(cfg.knn.k)
     
    total_iterations = total_iterations + 1
    # Initialize the progress bar
    with tqdm(total=total_iterations, desc="Combination of different hyperparameters") as pbar:
        for feat_type in cfg.knn.feature_type:
            for k in cfg.knn.k:
                for distance_fx in cfg.knn.distance_fx:
                    temperatures = cfg.knn.T if distance_fx == "cosine" else [None]
                    for T in temperatures:
                        acc1, acc5, confusion_matrix, recall, precision = run_knn(
                            train_features=train_features[feat_type],
                            train_targets=train_feats_tuple[1],
                            test_features=val_features[feat_type],
                            test_targets=val_feats_tuple[1],
                            k=k,
                            T=T,
                            distance_fx=distance_fx,
                        )
                        if acc1 > best_acc1:
                            best_acc1 = acc1
                            best_result = (feat_type, k, distance_fx, T, acc1, acc5, confusion_matrix, recall, precision)
                        
                        # Update the progress bar
                        pbar.update(1)
        


    if best_result:
        # run the best model on the test set
        feat_type, k, distance_fx, T, acc1, acc5, confusion_matrix, recall, precision = best_result
        acc1, acc5, confusion_matrix, recall, precision = run_knn(
            train_features=train_features[feat_type],
            train_targets=train_feats_tuple[1],
            test_features=test_features[feat_type],
            test_targets=test_feats_tuple[1],
            k=k,
            T=T,
            distance_fx=distance_fx,
        )
        best_result = (feat_type, k, distance_fx, T, acc1, acc5, confusion_matrix, recall, precision)
        #print best result
        #print(f"Best result: {best_result}")
        save_result_to_csv(best_result, cfg)


def save_result_to_csv(result, cfg):
    csv_file = cfg.output_csv
    if not csv_file.endswith(".csv"):
        csv_file += ".csv"

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a") as f:
        if not file_exists:
            f.write(
                "model_name,dataset,feature_type,k,distance_function,T,acc1,acc5,recall,precision\n"
            )
        
        f.write(
            f"{cfg.name},{cfg.data.dataset},{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]},{result[7]},{result[8]}\n"
        )


if __name__ == "__main__":
    main()
