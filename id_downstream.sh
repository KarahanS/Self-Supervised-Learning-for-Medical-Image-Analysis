#!/bin/bash

# Define constants
LINEAR_MEDMNIST_PATH=scripts/linear/medmnist/
MODEL_ROOT_DIR=/graphics/scratch2/students/kargibo/checkpoints/oguz  # CHANGE
CHECKPOINTS_DIR=/graphics/scratch2/students/kargibo/checkpoints/oguz/Downstream    # CHANGE
TRAIN_FRACTION=1.0 # CHANGE TO 0.01 FOR 1% AND 0.1 FOR 10% EXPERIMENT

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
    PRETRAINED_STATUS=$(echo $model_file | grep -o 'pretrained-[^-]*' | cut -d'-' -f2)

    if [[ $PRETRAINED_STATUS == "True" ]]; then
        PRETRAINED="pretrained"
    else
        PRETRAINED="not_pretrained"
    fi

    name="linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}"
    result_csv="downstream_results.csv"
    # if fraction of training data is not 1.0, then change the result_csv name
    if [[ $TRAIN_FRACTION != 1.0 ]]; then
        result_csv="downstream_results_${TRAIN_FRACTION}.csv"
    fi

    
    # check if the model that is about to be run is already run before from the downstream_results.csv using the model_name column
    if grep -q $name $result_csv; then
        echo "Model $name is already run before. Skipping..."
        continue
    fi

    # Construct the SSL checkpoint name
    SSL_CHECKPOINT_NAME="${model_dir#${MODEL_ROOT_DIR}/}"
    SSL_CHECKPOINT_NAME="${SSL_CHECKPOINT_NAME}/$model_file"
    
    echo "Running the downstream task for $name"
    
    # Run the downstream task
    python -W ignore::UserWarning main_linear.py --config-path $LINEAR_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
        pretrained_feature_extractor=${MODEL_ROOT_DIR}/${SSL_CHECKPOINT_NAME} \
        data="${DATASET_NAME}.yaml" \
        backbone.name=$ARCHITECTURE \
        backbone.kwargs.img_size=64 \
        downstream_classifier.name="linear" \
        checkpoint.dir=${CHECKPOINTS_DIR} \
        max_epochs=100 \
        name=${name} \
        data.train_fraction=${TRAIN_FRACTION} \
        to_csv.name=${result_csv} \
    
    # Erase leftover checkpoints
    rm -r ${CHECKPOINTS_DIR}/${name}*

done
