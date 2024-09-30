#!/bin/bash

# Add current path to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAMES=("simclr")

RESNET_MODEL="resnet50"

BATCH_SIZE=256
EPOCHS=400 
GRID_SEARCH_EPOCHS=$((EPOCHS / 4))
PRETRAINED=("False")
CHECKPOINT_DIR="/graphics/scratch2/students/saritas/imagenet_checkpoints"
CHECKPOINT_FILE="simclr_epoch_99-step_500400.ckpt" # there shouldn't be any '=' in the filename.

# List of datasets to run
DATASETS=("pathmnist" "dermamnist" "octmnist", "pneumoniamnist" "retinamnist" "breastmnist" "bloodmnist" "organamnist" "organcmnist" "organsmnist" "tissuemnist") # Replace with your actual dataset names
# DATASETS=("organamnist" "organcmnist" "organsmnist" "tissuemnist") 
# DATASETS=("pathmnist")

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
                auto_resume.enabled=True \
                name="selfsup_imagenet_init_${DATASET}-${METHOD_NAME}-${RESNET_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}" \
                checkpoint.dir=$CHECKPOINT_DIR checkpoint.filename=$CHECKPOINT_FILE
        done
    done
done