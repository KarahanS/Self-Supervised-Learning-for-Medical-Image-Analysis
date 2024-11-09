#!/bin/bash

# Add current path to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAMES=("simclr" "byol" "dino" "vicreg")


RESNET_MODEL="resnet50"

BATCH_SIZE=256
EPOCHS=400
GRID_SEARCH_EPOCHS=$((EPOCHS / 4))
PRETRAINED=("False")
CHECKPOINT_DIR="/graphics/scratch2/students/saritas/imagenet_checkpoints"

# Define load paths corresponding to each method

# Lightly: simclr, byol, dino, vicreg
# mmselfsup: mocov3
# all models are trained for 100 epoch
# simclr, byol, vicreg: 256 batch size
# dino: 128 batch size
# mocov3: 4096 batch size

# TODO: You can put all imagenet checkpoints to the same subfolder.  imagenet_checkpoints/checkpoints/
LOAD_PATHS=(
    "/graphics/scratch2/students/saritas/imagenet_checkpoints/in1k/simclr_epoch_99-step_500400.ckpt"  # for simclr
    "/graphics/scratch2/students/saritas/imagenet_checkpoints/in1k/byol_epoch_99-step_500400.ckpt"      # for byol
    "/graphics/scratch2/students/saritas/imagenet_checkpoints/in1k/dino_epoch_99-step_1000900.ckpt"     # for dino
    # "/graphics/scratch2/students/saritas/imagenet_checkpoints/in1k/vicreg_epoch_99-step_500400.ckpt"  # for vicreg
)

# List of datasets to run
DATASETS=("pneumoniamnist" "retinamnist" "breastmnist" "bloodmnist" "organamnist" "organcmnist" "organsmnist" "tissuemnist")
# "pathmnist" "dermamnist" "octmnist" <-- later


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
