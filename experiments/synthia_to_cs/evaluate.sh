#!/bin/bash

python3 ../../eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-type "vgg" \
        --model-name "eval_cs" \
        --model-load ubna_pretrained_models/ubna_pretrained_seg_synthia_vgg16/checkpoints/epoch_20 \
        --segmentation-validation-loaders "cityscapes_validation"

python3 ../../eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-type "vgg" \
        --model-name "eval_cs" \
        --model-load ubna_adapted_models/ubna_adaption_synthia_to_cs_vgg16_BS6_alpha_batch=0.08_alpha_layer=0/checkpoints/batch_50 \
        --segmentation-validation-loaders "cityscapes_validation"

python3 ../../eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-type "vgg" \
        --model-name "eval_cs" \
        --model-load ubna_adapted_models/ubna_adaption_synthia_to_cs_vgg16_BS6_alpha_batch=0.08_alpha_layer=0.03/checkpoints/batch_50 \
        --segmentation-validation-loaders "cityscapes_validation"
