#!/bin/bash

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
BYOL_RESNET_PATH=byol.yaml
BYOL_VIT_PATH=byol_vit.yaml

BATCH_SIZE=64
EPOCHS=800

DATASET_CFGS=("bloodmnist.yaml" "chestmnist.yaml")
EXPERIMENT_NAMES=("bloodmnist" "chestmnist")
PRETRAINED=("True" "False")


for PRETRAIN in "${PRETRAINED[@]}"
do
    for DATASET in "${DATASET_CFGS[@]}"
    do
        python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $BYOL_RESNET_PATH  \
            data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.kwargs.pretrained=$PRETRAIN \
            name="${DATASET}-r50-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}"
    done
done

for PRETRAIN in "${PRETRAINED[@]}"
    do
        for DATASET in "${DATASET_CFGS[@]}"
        do
        python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $BYOL_VIT_PATH \
                data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.kwargs.pretrained=$PRETRAIN \
            name="${DATASET}-vit-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}"

    done
done