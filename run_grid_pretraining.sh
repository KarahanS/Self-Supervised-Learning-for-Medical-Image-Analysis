#!/bin/bash

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAME=simclr
METHOD_RESNET_PATH=${METHOD_NAME}
METHOD_VIT_PATH=${METHOD_NAME}_vit

RESNET_MODEL="resnet50"
VIT_MODEL="vit_small"

BATCH_SIZE=256
EPOCHS=100 
GRID_SEARCH_EPOCHS=$((EPOCHS / 4))
DATASET_CFGS=("bloodmnist.yaml")
EXPERIMENT_NAMES=("bloodmnist")
PRETRAINED=("True")


for PRETRAIN in "${PRETRAINED[@]}"
do
    for DATASET in "${DATASET_CFGS[@]}"
    do
        python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_RESNET_PATH  \
            data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.name=$RESNET_MODEL backbone.kwargs.pretrained=$PRETRAIN \
            grid_search.enabled=True grid_search.linear_max_epochs=100 \
            grid_search.pretrain_max_epochs=$GRID_SEARCH_EPOCHS \
            name="${DATASET}-${METHOD_NAME}-${RESNET_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}" 
    
    done
done

# for PRETRAIN in "${PRETRAINED[@]}"
#     do
#         for DATASET in "${DATASET_CFGS[@]}"
#         do
#         python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_VIT_PATH \
#             data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
#             backbone.name=$VIT_MODEL backbone.kwargs.pretrained=$PRETRAIN \
#             name="${DATASET}-${METHOD_NAME}-${VIT_MODEL}-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}"

#     done
# done