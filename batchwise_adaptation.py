import copy
import numpy as np

import torch
import torch.nn as nn


class BatchWiseAdaptation(object):
    def __init__(self, train_mode_sequential, batch_size, encoder_batchnorm_momentum,
                 alpha_batch, alpha_layer):

        self.train_mode_sequential = train_mode_sequential

        self.batch_size = batch_size

        # determines the standard BN momentum
        self.encoder_batchnorm_momentum = encoder_batchnorm_momentum

        # Hyperparameter for our BN method to determine how fast
        # momentum is shrinking depending of num of batches already used for adaptation
        self.alpha_batch = alpha_batch

        # Hyperparameter for our BN method to determine how fast momentum is
        # shrinking depending of how deep the BN momentum is in the architecture
        self.alpha_layer = alpha_layer

    def process(self, model, batch_idx):

        if self.train_mode_sequential == 'none':
            return model

        # Our Approach: Shrinking BN momentum the more batches are trained
        elif self.train_mode_sequential == 'batch_shrinking':
            shared_encoder_bn_momentum = self.encoder_batchnorm_momentum * np.exp(-self.alpha_batch * batch_idx)
            # Check for architecture of encoder, because layers are accessed differently for different architectures
            if model._get_name() == 'UBNAVGG':
                for module in model.common.encoder.encoder.features.modules():
                    if type(module) == nn.BatchNorm2d:
                        module.momentum = shared_encoder_bn_momentum
            else:
                for module in model.common.encoder.encoder.modules():
                    if type(module) == nn.BatchNorm2d:
                        module.momentum = shared_encoder_bn_momentum

        # Our Approach: Shrinking BN momentum the more batches are trained.
        # And: The deeper the BN Layer the lower BN momentum
        elif self.train_mode_sequential == 'layer_shrinking':
            depth_of_bn_layer = 1
            shared_encoder_bn_momentum = self.encoder_batchnorm_momentum * np.exp(-self.alpha_batch * batch_idx)
            # Check for architecture of encoder, because layers are accessed differently for different architectures
            if model._get_name() == 'UBNAVGG':
                for module in model.common.encoder.encoder.features.modules():
                    if type(module) == nn.BatchNorm2d:
                        module.momentum = shared_encoder_bn_momentum * \
                                          np.exp(-self.alpha_layer * depth_of_bn_layer)
                        depth_of_bn_layer += 1
            else:
                for module in model.common.encoder.encoder.modules():
                    if type(module) == nn.BatchNorm2d:
                        module.momentum = shared_encoder_bn_momentum * \
                                          np.exp(-self.alpha_layer * depth_of_bn_layer)
                        depth_of_bn_layer += 1
        else:
            raise Exception('Unsupported train_mode_sequential specified')

        return model
