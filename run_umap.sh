
MODEL_CHECKPOINT_DIR="/graphics/scratch2/students/kargibo/tmp"

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
TRAIN_PATH="/graphics/scratch2/students/kargibo/dataset"
VAL_PATH="/graphics/scratch2/students/kargibo/dataset" 

METHOD_NAMES=("byol" "simclr" "mocov3")
DATASET_NAMES=("organamnist" "organcmnist" "organsmnist")
BACKBONE_NAMES=("resnet50" "vit_small")
TEST_DATASET_NAMES=("organamnist" "organcmnist" "organsmnist")

# Iterate over each folder in MODEL_CHECKPOINT_DIR
for METHOD_NAME in "${METHOD_NAMES[@]}"; do
    for MODEL_CHECKPOINT_NAME in "${MODEL_CHECKPOINT_DIR}/${METHOD_NAME}"/*; do

        # Parse dataset name from first - split
        # Get the folder name only
    
        CHECKPOINT_NAME=$(basename $MODEL_CHECKPOINT_NAME)
        DATASET_NAME=$(echo $CHECKPOINT_NAME | cut -d'-' -f1)
        DATASET_NAME=${DATASET_NAME%.yaml}

        BACKBONE_NAME=$(echo $CHECKPOINT_NAME | cut -d'-' -f3)

        if [[ ! " ${BACKBONE_NAMES[@]} " =~ " ${BACKBONE_NAME} " ]]; then
            echo "Skipping ${DATASET_NAME} as backbone name is not in ${BACKBONE_NAMES[@]}"
            continue
        fi

        echo "Running main_umap.py for model checkpoint: ${MODEL_CHECKPOINT_NAME}"

        for TEST_DATASET_NAME in "${TEST_DATASET_NAMES[@]}"; do
            echo "Running main_umap.py for test dataset: ${TEST_DATASET_NAME}"
            python main_umap.py --pretrained_checkpoint_dir "${MODEL_CHECKPOINT_NAME}" \
                --dataset ${TEST_DATASET_NAME} --train_data_path ${TRAIN_PATH} --val_data_path ${VAL_PATH} --data_format image_folder
        done
    done
done