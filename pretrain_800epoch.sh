#!/bin/bash

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAME=simclr
METHOD_RESNET_PATH=${METHOD_NAME}.yaml
METHOD_VIT_PATH=${METHOD_NAME}_vit.yaml

RESNET_MODEL="resnet50"
VIT_MODEL="vit_tiny"

BATCH_SIZE=256
EPOCHS=800

DATASET_CFGS=("dermamnist.yaml", "pneumoniamnist.yaml", "retinamnist.yaml", "breastmnist.yaml", "bloodmnist.yaml", "organamnist.yaml", "organcmnist.yaml", "organsmnist.yaml")
EXPERIMENT_NAMES=("dermamnist", "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist", "organamnist", "organcmnist", "organsmnist")
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
