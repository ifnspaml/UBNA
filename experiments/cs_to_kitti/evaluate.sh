#!/bin/bash
        
python3 ../../eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-type "vgg" \
        --model-name "eval_kitti" \
        --model-load ubna_pretrained_models/ubna_pretrained_seg_cs_vgg16/checkpoints/epoch_20 \
        --segmentation-validation-loaders "kitti_2015_train" \
        --segmentation-validation-resize-width 1024 \
        --segmentation-validation-resize-height 320
        
python3 ../../eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-type "vgg" \
        --model-name "eval_kitti" \
        --model-load ubna_adapted_models/ubna_adaption_cs_to_kitti_vgg16_BS6_alpha_batch=0.08_alpha_layer=0/checkpoints/batch_50 \
        --segmentation-validation-loaders "kitti_2015_train" \
        --segmentation-validation-resize-width 1024 \
        --segmentation-validation-resize-height 320
        
python3 ../../eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-type "vgg" \
        --model-name "eval_kitti" \
        --model-load ubna_adapted_models/ubna_adaption_cs_to_kitti_vgg16_BS6_alpha_batch=0.08_alpha_layer=0.3/checkpoints/batch_50 \
        --segmentation-validation-loaders "kitti_2015_train" \
        --segmentation-validation-resize-width 1024 \
        --segmentation-validation-resize-height 320
