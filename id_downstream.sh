#!/bin/bash

# Define constants
LINEAR_MEDMNIST_PATH=scripts/linear/medmnist/
MODEL_ROOT_DIR=/graphics/scratch2/students/kargibo/checkpoints/oguz  # CHANGE
CHECKPOINTS_DIR=/graphics/scratch2/students/kargibo/checkpoints/oguz/Downstream    # CHANGE

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
    

    # check if the model that is about to be run is already run before from the downstream_results.csv using the model_name column
    # using name="linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}" as the model_name
    if grep -q "linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}" downstream_results.csv; then
        echo "Model linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED} is already run."
        continue
    fi

    
    # Construct the SSL checkpoint name
    SSL_CHECKPOINT_NAME="${model_dir#${MODEL_ROOT_DIR}/}"
    SSL_CHECKPOINT_NAME="${SSL_CHECKPOINT_NAME}/$model_file"
    
    # Erase the content of the CHECKPOINTS_DIR
    rm -rf ${CHECKPOINTS_DIR}/*
    
    # Run the downstream task
    python main_linear.py --config-path $LINEAR_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
        pretrained_feature_extractor=${MODEL_ROOT_DIR}/${SSL_CHECKPOINT_NAME} \
        data="${DATASET_NAME}.yaml" \
        backbone.name=$ARCHITECTURE \
        backbone.kwargs.img_size=64 \
        downstream_classifier.name="linear" \
        checkpoint.dir=${CHECKPOINTS_DIR} \
        max_epochs=100 \
        name="linear-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}"
        
done
