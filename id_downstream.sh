#!/bin/bash

# Define constants
LINEAR_MEDMNIST_PATH=scripts/linear/medmnist/
MODEL_ROOT_DIR=/graphics/scratch2/students/saritas/trained_models

# Recursively search for models ending with 'best_val_acc1.ckpt'
find $MODEL_ROOT_DIR -type f -name "*best_val_acc1.ckpt" | while read -r model_path; do

    # Extract the directory path and model file name
    model_dir=$(dirname "$model_path")
    model_file=$(basename "$model_path")
    
    # Extract the METHOD_NAME and DATASET_NAME from the model file name
    DATASET_NAME=$(echo $model_file | cut -d'-' -f1 | cut -d'.' -f1)
    METHOD_NAME=$(echo $model_file | cut -d'-' -f2)
    
    # Construct the SSL checkpoint name
    SSL_CHECKPOINT_NAME="${model_dir#${MODEL_ROOT_DIR}/}"
    SSL_CHECKPOINT_NAME="${SSL_CHECKPOINT_NAME}/$model_file"
    
    # Define the downstream classifier types
    for classifier in "linear" "mlp"; do
        # Run the downstream task
        python main_linear.py --config-path $LINEAR_MEDMNIST_PATH --config-name ${METHOD_NAME}.yaml \
            pretrained_feature_extractor=${MODEL_ROOT_DIR}/${SSL_CHECKPOINT_NAME} \
            data="${DATASET_NAME}.yaml"  \
            downstream_classifier.name="${classifier}" \
            max_epochs=100 \
            name="${classifier}-${DATASET_NAME}-${METHOD_NAME}" \
        
    done
done