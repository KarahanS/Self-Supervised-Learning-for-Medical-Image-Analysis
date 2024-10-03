#!/bin/bash

# Add current path to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAMES=("simclr" "byol")

RESNET_MODEL="resnet50"

BATCH_SIZE=256
EPOCHS=400
GRID_SEARCH_EPOCHS=$((EPOCHS / 4))
PRETRAINED=("False")
CHECKPOINT_DIR="/graphics/scratch2/students/saritas/imagenet_checkpoints"

# Define load paths corresponding to each method
LOAD_PATHS=(
    "/graphics/scratch2/students/saritas/imagenet_checkpoints/imagenet_simclr/simclr_epoch_99-step_500400.ckpt"  # for simclr
    "/graphics/scratch2/students/saritas/imagenet_checkpoints/imagenet_byol/byol_epoch_99-step_500400.ckpt"    # for byol
)

# List of datasets to run
DATASETS=("pathmnist" "dermamnist" "octmnist" "pneumoniamnist" "retinamnist" "breastmnist" "bloodmnist" "organamnist" "organcmnist" "organsmnist" "tissuemnist")

# Loop over datasets
for DATASET in "${DATASETS[@]}"
do
    echo "Running experiments for dataset: $DATASET"

    for i in "${!METHOD_NAMES[@]}"
    do
        METHOD_NAME=${METHOD_NAMES[$i]}
        LOAD_PATH=${LOAD_PATHS[$i]}  # Use the corresponding load path for the method

        for PRETRAIN in "${PRETRAINED[@]}"
        do
            python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_NAME  \
                data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
                backbone.name=$RESNET_MODEL backbone.kwargs.pretrained=$PRETRAIN \
                grid_search.enabled=True grid_search.linear_max_epochs=100 \
                grid_search.pretrain_max_epochs=$GRID_SEARCH_EPOCHS \
                auto_resume.enabled=False \
                load.enabled=True \
                load.path=$LOAD_PATH \
                name="selfsup_imagenet_init_${DATASET}-${METHOD_NAME}-${RESNET_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}" \
                checkpoint.dir=$CHECKPOINT_DIR 
        done
    done
done
