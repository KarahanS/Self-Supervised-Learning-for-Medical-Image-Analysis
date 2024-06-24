#!/bin/bash

# Define constants
KNN_MEDMNIST_PATH=scripts/knn/medmnist/
MODEL_ROOT_DIR=/graphics/scratch2/students/kargibo/checkpoints/oguz  # CHANGE
OUTPUT_CSV=knn_results.csv

K="[1,2,5,10,20,50,100,200]"
T="[0.07,0.1,0.2,0.5,1.0,2.0]"
FEATURE_TYPE="[backbone]"
DISTANCE_FX="[euclidean,cosine]"

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

    # if the checkpoint name contains "pretrained-True" or "pretrained-False"
    if [[ $PRETRAINED_STATUS == "True" ]]; then
        PRETRAINED="pretrained"
    else
        PRETRAINED="not_pretrained"
    fi

    # Check if the model that is about to be run is already run before from the knn_results.csv using the model_name column
    # using name="knn-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}" as the model_name
    if grep -q "knn-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}" $OUTPUT_CSV; then
        echo "Model knn-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED} is already run."
        continue
    fi
    
    # Construct the SSL checkpoint name
    SSL_CHECKPOINT_NAME="${model_dir#${MODEL_ROOT_DIR}/}"
    SSL_CHECKPOINT_NAME="${SSL_CHECKPOINT_NAME}/$model_file"
    
    # Run the knn task
    python main_knn.py \
        --config-path $KNN_MEDMNIST_PATH \
        --config-name ${METHOD_NAME}.yaml \
        pretrained_feature_extractor=${MODEL_ROOT_DIR}/${SSL_CHECKPOINT_NAME} \
        pretrain_method=$METHOD_NAME \
        data.dataset=$DATASET_NAME \
        data.image_size=64 \
        knn.k=$K \
        knn.T=$T \
        knn.feature_type=$FEATURE_TYPE \
        knn.distance_fx=$DISTANCE_FX \
        output_csv=$OUTPUT_CSV \
        backbone.name=$ARCHITECTURE \
        backbone.kwargs.img_size=64 \
        name="knn-${DATASET_NAME}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}"

done
