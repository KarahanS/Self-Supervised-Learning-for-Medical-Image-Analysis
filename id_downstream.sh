#!/bin/bash

## Important: If you stop the script before all seeds are finished, this script doesnt know how many seeds are supposed to run, so it will skip even if only 1 seed is finished.

# Define constants
LINEAR_MEDMNIST_PATH=scripts/linear/medmnist/
TRAIN_FRACTION=1.0 # CHANGE TO 0.01 FOR 1% AND 0.1 FOR 10% EXPERIMENT
SEEDS=5 # parameter that controls how many different seeds are tried for downstream


# List of model root directories to loop over
MODEL_ROOT_DIRS=(
    "/graphics/scratch2/students/kargibo/checkpoints/oguz"
    "/graphics/scratch2/students/kargibo/checkpoints/kivanc/trained_models"
    "/graphics/scratch2/students/saritas/trained_models"  # Add more directories as needed
    "/graphics/scratch2/students/kargibo/experiments"
)

# Loop over each model root directory
for MODEL_ROOT_DIR in "${MODEL_ROOT_DIRS[@]}"; do
    CHECKPOINTS_DIR="/graphics/scratch2/students/kargibo/checkpoints/oguz/Downstream"  # Update the checkpoints directory for each root
    # append the fraction to the checkpoints directory
    CHECKPOINTS_DIR="${CHECKPOINTS_DIR}_fraction_${TRAIN_FRACTION}"

    # Recursively search for models ending with 'best_val_acc1.ckpt'
    find $MODEL_ROOT_DIR -type f -name "*best_val_acc1.ckpt" | while read -r model_path; do

        # Extract the directory path and model file name
        model_dir=$(dirname "$model_path")
        model_file=$(basename "$model_path")

        # Extract the DATASET_NAME, METHOD_NAME, ARCHITECTURE, and whether it is pretrained
        DATASET_NAME=$(echo $model_file | cut -d'-' -f1 | cut -d'.' -f1)
        METHOD_NAME=$(echo $model_file | cut -d'-' -f2)
        ARCHITECTURE=$(echo $model_file | cut -d'-' -f3)
        # Extract the pretrained status using a more flexible method to accommodate various naming conventions
        if echo "$model_file" | grep -q 'pretrained-False'; then
            PRETRAINED_STATUS="False"
        elif echo "$model_file" | grep -q 'pretrained-True'; then
            PRETRAINED_STATUS="True"
        elif echo "$model_file" | grep -q 'not_pretrained'; then
            PRETRAINED_STATUS="False"
        else
            PRETRAINED_STATUS="True"
        fi

        if [[ $PRETRAINED_STATUS == "True" ]]; then
            PRETRAINED="pretrained"
        else
            PRETRAINED="not_pretrained"
        fi

        # Filter such that only methods simclr, byol, or mocov3 are considered
        if [[ ! ($METHOD_NAME == "simclr" || $METHOD_NAME == "byol" || $METHOD_NAME == "mocov3") ]]; then
            continue
        fi

        # Filter such that only resnet50, not pretrained, and simclr, byol, or mocov3 models are considered
        # if [[ $ARCHITECTURE != "resnet50" || $PRETRAINED == "pretrained" || ! ($METHOD_NAME == "simclr" || $METHOD_NAME == "byol" || $METHOD_NAME == "mocov3") ]]; then
        #     continue
        # fi

        name="linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}"
        result_csv="paper_downstream_results.csv"

        # if fraction of training data is not 1.0, then change the name of the saved model
        if [[ $TRAIN_FRACTION != 1.0 ]]; then
            name="linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}-fraction_${TRAIN_FRACTION}"
            result_csv="paper_downstream_results_${TRAIN_FRACTION}.csv"
        fi

        # check if the model that is about to be run is already run before from the downstream_results.csv using the model_name column
        if grep -q $name $result_csv; then
            echo "Model $name is already run before. Skipping..."
            continue
        fi

        # Construct the SSL checkpoint name
        SSL_CHECKPOINT_NAME="${model_dir#${MODEL_ROOT_DIR}/}"
        SSL_CHECKPOINT_NAME="${SSL_CHECKPOINT_NAME}/$model_file"

        # Erase the content of the CHECKPOINTS_DIR
        rm -rf ${CHECKPOINTS_DIR}/*

        echo "Running the downstream task for $name"

        # Run the downstream task
        python -W ignore::UserWarning main_linear.py --config-path $LINEAR_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
            pretrained_feature_extractor=${MODEL_ROOT_DIR}/${SSL_CHECKPOINT_NAME} \
            data="${DATASET_NAME}.yaml" \
            backbone.name=$ARCHITECTURE \
            backbone.kwargs.img_size=64 \
            downstream_classifier.name="linear" \
            downstream_classifier.kwargs.num_seeds=${SEEDS} \
            checkpoint.dir=${CHECKPOINTS_DIR} \
            max_epochs=100 \
            name=${name} \
            data.train_fraction=${TRAIN_FRACTION} \
            to_csv.name=${result_csv} \

    done
done
