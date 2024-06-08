#!/bin/bash

PRETRAIN_MEDMNIST_PATH=scripts/pretrain/medmnist/
METHOD_NAME=byol
METHOD_RESNET_PATH=${METHOD_NAME}.yaml
METHOD_VIT_PATH=${METHOD_NAME}_vit.yaml

RESNET_MODEL="resnet50"
VIT_MODEL="vit_tiny"

# Validate the script by running it with a smaller number of epochs
BATCH_SIZE=32
EPOCHS=1

DATASET_CFGS=("chestmnist.yaml")
EXPERIMENT_NAMES=("chestmnist")
PRETRAINED=("True" "False")



for PRETRAIN in "${PRETRAINED[@]}"
do
    for DATASET in "${DATASET_CFGS[@]}"
    do

    
        echo "Running: python main_solo.py --config-path ${PRETRAIN_MEDMNIST_PATH} --config-name ${METHOD_RESNET_PATH}  \
            data=${DATASET} optimizer.batch_size=${BATCH_SIZE} max_epochs=${EPOCHS} \
            backbone.name=${RESNET_MODEL} backbone.kwargs.pretrained=${PRETRAIN} \
            name='_validate-${DATASET}-r50-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}'"
        python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_RESNET_PATH  \
            data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.name=$RESNET_MODEL backbone.kwargs.pretrained=$PRETRAIN \
            name="_validate-${DATASET}-r50-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}"
    done
done

for PRETRAIN in "${PRETRAINED[@]}"
    do
        for DATASET in "${DATASET_CFGS[@]}"
        do
        echo "Running: python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_VIT_PATH \
            data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.name=$VIT_MODEL backbone.kwargs.pretrained=$PRETRAIN \
            name='_validate-${DATASET}-vit-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}'"
            
        python main_solo.py --config-path $PRETRAIN_MEDMNIST_PATH --config-name $METHOD_VIT_PATH \
            data=$DATASET optimizer.batch_size=$BATCH_SIZE max_epochs=$EPOCHS \
            backbone.name=$VIT_MODEL backbone.kwargs.pretrained=$PRETRAIN \
            name="_validate-${DATASET}-vit-epochs-${EPOCHS}-pretrained-${PRETRAIN}-batch_size-${BATCH_SIZE}"

    done
done
echo "Validation complete"