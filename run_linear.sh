#!/bin/bash

LINEAR_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAME=byol
DATASET_NAME="bloodmnist"
BACKBONE="resnet50"
MODEL_CHECKPOINTS=trained_models"
SSL_CHECKPOINT_NAME="fwgkvt8s/byol-custom-dataset-fwgkvt8s-ep-0.ckpt"

python main_linear.py --config-path $LINEAR_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
        pretrained_feature_extractor=${MODEL_CHECKPOINTS}/${SSL_CHECKPOINT_NAME} \
        data="bloodmnist.yaml"  \
        downstream_classifier.name="linear" \
        name="${DATASET_NAME}-${METHOD_NAME}-r50-epochs-100-pretrained-True-batch_size-64" 