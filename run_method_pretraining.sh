#!/bin/bash

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAME=dino
METHOD_RESNET_PATH=${METHOD_NAME}.yaml
METHOD_VIT_PATH=${METHOD_NAME}_vit.yaml

RESNET_MODEL="resnet50"
VIT_MODEL="vit_tiny"

BATCH_SIZE=256
EPOCHS=800

DATASET_CFGS=("bloodmnist.yaml" "chestmnist.yaml" "pathmnist.yaml")
EXPERIMENT_NAMES=("bloodmnist" "chestmnist" "pathmnist")
PRETRAINED=("True" "False")


for PRETRAIN in "${PRETRAINED[@]}"
do
    for DATASET in "${DATASET_CFGS[@]}"
    do
        python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_RESNET_PATH  \
            data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.name=$RESNET_MODEL backbone.kwargs.pretrained=$PRETRAIN \
            name="${DATASET}-${METHOD_NAME}-${RESNET_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}"
    
    done
done

for PRETRAIN in "${PRETRAINED[@]}"
    do
        for DATASET in "${DATASET_CFGS[@]}"
        do
        python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_VIT_PATH \
            data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.name=$VIT_MODEL backbone.kwargs.pretrained=$PRETRAIN \
            name="${DATASET}-${METHOD_NAME}-${VIT_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}"

    done
done
