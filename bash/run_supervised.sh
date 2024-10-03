#!/bin/bash

# Add current path to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

PRETRAIN_MEDMNIST_PATH=scripts/linear/medmnist/
RESNET_MODEL="resnet50"
VIT_MODEL="vit_small"
METHOD_NAMES=("supervised")
BATCH_SIZE=256
TRAIN_FRACTION=1.0
SEEDS=5
EPOCHS=100
GRID_SEARCH_EPOCHS=$((EPOCHS / 4))
PRETRAINED=("True")
CHECKPOINT_DIR="/graphics/scratch2/students/saritas/supervised"

# List of datasets to run
DATASETS=("tissuemnist" "pathmnist" "dermamnist" "octmnist" "pneumoniamnist" "retinamnist" "breastmnist" "bloodmnist" "organamnist" "organcmnist" "organsmnist") # Removed the extra comma
# DATASETS=("organamnist" "organcmnist" "organsmnist") 

# Loop over datasets
for DATASET in "${DATASETS[@]}"
do
    echo "Running experiments for dataset: $DATASET"

    for METHOD_NAME in "${METHOD_NAMES[@]}"
    do
        for PRETRAIN in "${PRETRAINED[@]}"
        do
            # Dataset name variable was incorrect
            name="supervised-${DATASET}-${RESNET_MODEL}-${PRETRAIN}"
            result_csv="supervised_results.csv"

            # if fraction of training data is not 1.0, then change the name of the saved model
            if [[ $TRAIN_FRACTION != 1.0 ]]; then
                name="supervised-${DATASET}-${RESNET_MODEL}-${PRETRAIN}-fraction_${TRAIN_FRACTION}"
                result_csv="supervised_results_${TRAIN_FRACTION}.csv"
            fi

            # Run the supervised training
            python  main_linear.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
                data="${DATASET}.yaml" \
                backbone.name=$RESNET_MODEL \
                downstream_classifier.name="linear" \
                backbone.kwargs.pretrained=$PRETRAIN \
                downstream_classifier.kwargs.num_seeds=${SEEDS} \
                checkpoint.dir=${CHECKPOINT_DIR} \
                max_epochs=${EPOCHS} \
                name=${name} \
                data.train_fraction=${TRAIN_FRACTION} \
                to_csv.name=${result_csv}
        done
    done
done
