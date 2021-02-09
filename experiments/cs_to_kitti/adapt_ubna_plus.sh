#!/usr/bin/env bash

python3 ../../adapt.py \
        --model-name ubna_adaption_cs_to_kitti_vgg16_BS6_alpha_batch=0.08_alpha_layer=0.3 \
        --adaptation-training-loaders "kitti_kitti_train" \
        --adaptation-resize-width 1024 \
        --adaptation-resize-height 320 \
        --adaptation-training-batch-size 6 \
        --model-type "vgg" \
        --model-num-layers-vgg 16 \
        --model-load ubna_pretrained_models/ubna_pretrained_seg_cs_vgg16/checkpoints/epoch_20 \
        --model-disable-lr-loading \
        --experiment-class "ubna_adapted_models" \
        --adaptation-alpha-batch 0.08 \
        --adaptation-alpha-layer 0.3 \
        --adaptation-mode-sequential "layer_shrinking" \
        --adaptation-num-batches 50
