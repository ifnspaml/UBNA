#!/usr/bin/env bash

python3 ../../train.py \
        --model-name ubna_pretrained_seg_gta_vgg16 \
        --model-type "vgg" \
        --model-num-layers-vgg 16 \
        --segmentation-training-loaders "gta5_train" \
        --segmentation-training-batch-size 12 \
        --experiment-class "ubna_pretrained_models"
