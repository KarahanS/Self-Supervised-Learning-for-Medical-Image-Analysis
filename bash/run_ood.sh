#!/bin/bash

OOD_MEDMNIST_PATH=scripts/ood/medmnist/
MODEL_ROOT_DIR="/graphics/scratch2/students/kargibo/experiments"
METHOD_NAMES=("simclr" "byol" "dino" "ressl" "mocov3")
export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASETS=("organamnist" "organcmnist" "organsmnist" "pathmnist" "tissuemnist" "breastmnist" "pneumoniamnist" "dermamnist" "retinamnist" "octmnist")
# Recursively search for models ending with 'best_val_acc1.ckpt'
for ID_DATASET in "${DATASETS[@]}"; do
    for OOD_DATASET in "${DATASETS[@]}"; do
        if [[ $ID_DATASET == $OOD_DATASET ]]; then
            continue
        fi

        for METHOD_NAME in "${METHOD_NAMES[@]}"; do
            find $MODEL_ROOT_DIR/$METHOD_NAME -type f -name "*curr-ep*" | while read -r model_path; do
                
                # Extract the directory path and model file name
                model_dir=$(dirname "$model_path")
                model_file=$(basename "$model_path")
                
                # Extract the DATASET_NAME, METHOD_NAME, ARCHITECTURE, and whether it is pretrained
                DATASET_NAME=$(echo $model_file | cut -d'-' -f1 | cut -d'.' -f1)
                METHOD_NAME=$(echo $model_file | cut -d'-' -f2)
                ARCHITECTURE=$(echo $model_file | cut -d'-' -f3)
                
            
                # If dataset names does not match skip
                if [[ $DATASET_NAME != $ID_DATASET ]]; then
                    continue
                fi

                # If architecture has vit in it, skip
                if [[ $ARCHITECTURE == *"vit"* ]]; then
                    continue
                fi

                if [[ $PRETRAINED_STATUS == "True" ]]; then
                    PRETRAINED="pretrained"
                else
                    PRETRAINED="not_pretrained"
                fi
                
                # Construct the SSL checkpoint name
                SSL_CHECKPOINT_NAME="${model_dir#${MODEL_ROOT_DIR}/}"
                SSL_CHECKPOINT_NAME="${SSL_CHECKPOINT_NAME}/$model_file"
                
                echo "Running the downstream task for $DATASET_NAME with $METHOD_NAME on $ARCHITECTURE"
                echo "SSL checkpoint: $SSL_CHECKPOINT_NAME"

                python main_ood.py \
                    --config-path $OOD_MEDMNIST_PATH \
                    --config-name ${METHOD_NAME} \
                    pretrained_feature_extractor=${MODEL_ROOT_DIR}/${SSL_CHECKPOINT_NAME} \
                    pretrain_method=$METHOD_NAME \
                    data.dataset=$ID_DATASET \
                    ood_data.dataset=$OOD_DATASET \
                    backbone.name=$ARCHITECTURE \
                    backbone.kwargs.img_size=64 \
                    wandb.enabled=False \
                    name="ood-${OOD_DATASET}-${METHOD_NAME}-${ARCHITECTURE}-${PRETRAINED}" \
                    to_csv.name="last_mahob_ood_results.csv"
            done
        done
    done
done