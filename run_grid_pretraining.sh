#!/bin/bash

# Add current path to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAMES=("simclr" "byol" "dino" "ressl" "mocov3")

RESNET_MODEL="vit_small"

BATCH_SIZE=256
EPOCHS=400
GRID_SEARCH_EPOCHS=$((EPOCHS / 4))
PRETRAINED=("False" "True")
CHECKPOINT_DIR="/graphics/scratch2/students/kargibo/experiments"

# Get the dataset from argument
DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Please provide a dataset as an argument"
    exit 1
fi

for PRETRAIN in "${PRETRAINED[@]}"
do
    for METHOD_NAME in "${METHOD_NAMES[@]}"
    do
            # Check if the RESNET_MODEL starts with "vit"
            if [[ $RESNET_MODEL == vit* ]]; then
                CONFIG_NAME="${METHOD_NAME}_vit"
            else
                CONFIG_NAME=$METHOD_NAME
            fi

            echo "Running $CONFIG_NAME on $DATASET with $RESNET_MODEL backbone"
            python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $CONFIG_NAME  \
                data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
                backbone.name=$RESNET_MODEL backbone.kwargs.pretrained=$PRETRAIN \
                grid_search.enabled=True grid_search.linear_max_epochs=100 \
                grid_search.pretrain_max_epochs=$GRID_SEARCH_EPOCHS \
                name="${DATASET}-${METHOD_NAME}-${RESNET_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}" \
                checkpoint.dir=$CHECKPOINT_DIR
    done
done
