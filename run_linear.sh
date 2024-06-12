#!/bin/bash

LINEAR_MEDMNIST_PATH=scripts/linear/medmnist/
METHOD_NAME=simclr
DATASET_NAME=pathmnist
BACKBONE=resnet50
MODEL_CHECKPOINTS=trained_models
SSL_CHECKPOINT_NAME="simclr/pathmnist.yaml-simclr-resnet50-epochs-10-pretrained-True-batch_size-256-7uapkt9u/pathmnist.yaml-simclr-resnet50-epochs-10-pretrained-True-batch_size-256-best_val_acc1.ckpt"

python main_linear.py --config-path $LINEAR_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
        pretrained_feature_extractor=${MODEL_CHECKPOINTS}/${SSL_CHECKPOINT_NAME} \
        data="${DATASET_NAME}.yaml"  \
        downstream_classifier.name="linear" \
        max_epochs=2 \
        name="linear-${DATASET_NAME}-${METHOD_NAME}-r50-epochs-100-pretrained-batch_size-64" \
