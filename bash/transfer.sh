#!/bin/bash

# Define constants
LINEAR_MEDMNIST_PATH=scripts/linear/medmnist/
TRAIN_FRACTION=1.0 # CHANGE TO 0.01 FOR 1% AND 0.1 FOR 10% EXPERIMENT
SEEDS=5 # parameter that controls how many different seeds are tried for downstream

# List of datasets to loop over
DATASETS=("pathmnist" "dermamnist" "octmnist" "pneumoniamnist" "retinamnist" "breastmnist" "bloodmnist" "organamnist" "organcmnist" "organsmnist" "tissuemnist")

# List of model root directories to loop over
MODEL_ROOT_DIRS=(
    "/graphics/scratch2/students/kargibo/experiments"
)

# Loop over each model root directory
for MODEL_ROOT_DIR in "${MODEL_ROOT_DIRS[@]}"; do
    CHECKPOINTS_DIR="/graphics/scratch2/students/saritas/downstream"  # Update the checkpoints directory for each root
    # append the fraction to the checkpoints directory
    CHECKPOINTS_DIR="${CHECKPOINTS_DIR}_fraction_${TRAIN_FRACTION}"

    # Recursively search for models that contain 'curr-ep' in their name
    for model_path in $(find $MODEL_ROOT_DIR -type f -name "*curr-ep*"); do

        # Extract the directory path and model file name
        model_dir=$(dirname "$model_path")
        model_file=$(basename "$model_path")

        # Extract the DATASET_NAME, METHOD_NAME, ARCHITECTURE, and whether it is pretrained
        DATASET_NAME=$(echo $model_file | cut -d'-' -f1 | cut -d'.' -f1)
        METHOD_NAME=$(echo $model_file | cut -d'-' -f2)
        ARCHITECTURE=$(echo $model_file | cut -d'-' -f3)

        # skip if it is not resnet50
        #if [[ $ARCHITECTURE == "resnet50" ]]; then
        #    continue
        #fi

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

        name="linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}"
        result_csv="transfer_downstream_results.csv"

        # if fraction of training data is not 1.0, then change the name of the saved model
        if [[ $TRAIN_FRACTION != 1.0 ]]; then
            name="linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}-fraction_${TRAIN_FRACTION}"
            result_csv="transfer_downstream_results_${TRAIN_FRACTION}.csv"
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

        # Test the model on all datasets except the one it was trained on
        for TEST_DATASET in "${DATASETS[@]}"; do
            if [[ $TEST_DATASET == $DATASET_NAME ]]; then
                echo "Skipping the original training dataset ($TEST_DATASET) for model $name..."
                continue
            fi

            echo "Testing $name on $TEST_DATASET..."

            # Run the downstream task on the current test dataset
            python -W ignore::UserWarning main_linear.py --config-path $LINEAR_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
                pretrained_feature_extractor=${MODEL_ROOT_DIR}/${SSL_CHECKPOINT_NAME} \
                data="${TEST_DATASET}.yaml" \
                backbone.name=$ARCHITECTURE \
                backbone.kwargs.img_size=64 \
                downstream_classifier.name="linear" \
                downstream_classifier.kwargs.num_seeds=${SEEDS} \
                checkpoint.dir=${CHECKPOINTS_DIR} \
                max_epochs=100 \
                name="${name}_transfer_${TEST_DATASET}" \
                data.train_fraction=${TRAIN_FRACTION} \
                to_csv.name=${result_csv} \
            
        done
    done
done
