#!/bin/bash

# Add current path to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAMES=("simclr" "byol" "dino" "ressl" "mocov3")

RESNET_MODEL="resnet50"

BATCH_SIZE=256
EPOCHS=400
GRID_SEARCH_EPOCHS=$((EPOCHS / 4))
PRETRAINED=("True")
CHECKPOINT_DIR="/graphics/scratch2/students/saritas/checkpoints"

# List of datasets to run
# DATASETS=("pathmnist" "dermamnist" "octmnist", "pneumoniamnist" "retinamnist" "breastmnist" "bloodmnist") # Replace with your actual dataset names
DATASETS=("organamnist" "organcmnist" "organsmnist" "tissuemnist") 

# Loop over datasets
for DATASET in "${DATASETS[@]}"
do
    echo "Running experiments for dataset: $DATASET"

    for METHOD_NAME in "${METHOD_NAMES[@]}"
    do
        for PRETRAIN in "${PRETRAINED[@]}"
        do
            python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_NAME  \
                data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
                backbone.name=$RESNET_MODEL backbone.kwargs.pretrained=$PRETRAIN \
                grid_search.enabled=True grid_search.linear_max_epochs=100 \
                grid_search.pretrain_max_epochs=$GRID_SEARCH_EPOCHS \
                name="${DATASET}-${METHOD_NAME}-${RESNET_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}" \
                checkpoint.dir=$CHECKPOINT_DIR
        done
    done
done
