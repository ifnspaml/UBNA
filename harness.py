#!/usr/bin/env python3

# Python standard library
import json
import os

# Public libraries
import torch
import torch.nn.functional as functional

# Local imports
import dataloader.file_io.get_path as get_path
import loaders, loaders.segmentation, loaders.adaptation
from losses.segmentation import RemappingScore
from state_manager import StateManager


class Harness(object):
    def __init__(self, opt):
        print('Starting initialization', flush=True)

        self._init_device(opt)
        self._init_losses(opt)
        self._init_log_dir(opt)
        self._init_logging(opt)
        self._init_tensorboard(opt)
        self._init_state(opt)
        self._init_train_loaders(opt)
        self._init_training(opt)
        self._init_validation(opt)
        self._init_validation_loaders(opt)
        self._save_opts(opt)

        print('Summary:')
        print(f'  - Model name: {opt.model_name}')
        print(f'  - Logging directory: {self.log_path}')
        print(f'  - Using device: {self._pretty_device_name()}')

    def _init_device(self, opt):
        cpu = not torch.cuda.is_available()
        cpu = cpu or opt.sys_cpu

        self.device = torch.device("cpu" if cpu else "cuda")

    def _init_losses(self, opt):
        pass

    def _init_validation(self, opt):
        pass

    def _init_log_dir(self, opt):
        path_getter = get_path.GetPath()
        log_base = path_getter.get_checkpoint_path()

        self.log_path = os.path.join(log_base, opt.experiment_class, opt.model_name)

        os.makedirs(self.log_path, exist_ok=True)

    def _init_logging(self, opt):
        pass

    def _init_tensorboard(self, opt):
        pass

    def _init_state(self, opt):
        self.state = StateManager(
            opt.experiment_class, opt.model_name, self.device,
            opt.model_type, opt.model_num_layers, opt.model_num_layers_vgg,
            opt.train_weights_init, opt.train_learning_rate, opt.train_weight_decay, opt.train_scheduler_step_size
        )
        if opt.model_load is not None:
            self.state.load(opt.model_load, opt.model_disable_lr_loading)

    def _init_train_loaders(self, opt):
        pass

    def _init_training(self, opt):
        pass

    def _init_validation_loaders(self, opt):
        print('Loading validation dataset metadata:', flush=True)

        # if the number of validation loaders is larger than 1, use standard resize values
        num_validation_loaders = len(opt.segmentation_validation_loaders.split(','))
        standard_resize = {'cityscapes_validation': (512, 1024),
                           'kitti_2015_train': (320, 1024)}

        if num_validation_loaders > 1:
            if hasattr(opt, 'segmentation_validation_loaders'):
                self.segmentation_validation_loader = loaders.ChainedLoaderList(
                    getattr(loaders.segmentation, loader_name)(
                        resize_height=standard_resize[loader_name][0],
                        resize_width=standard_resize[loader_name][1],
                        batch_size=opt.segmentation_validation_batch_size,
                        num_workers=opt.sys_num_workers
                    )
                    for loader_name in opt.segmentation_validation_loaders.split(',') if (loader_name != '')
                )
        else:
            if hasattr(opt, 'segmentation_validation_loaders'):
                self.segmentation_validation_loader = loaders.ChainedLoaderList(
                    getattr(loaders.segmentation, loader_name)(
                        resize_height=opt.segmentation_validation_resize_height,
                        resize_width=opt.segmentation_validation_resize_width,
                        batch_size=opt.segmentation_validation_batch_size,
                        num_workers=opt.sys_num_workers
                    )
                    for loader_name in opt.segmentation_validation_loaders.split(',') if (loader_name != '')
                )

    def _pretty_device_name(self):
        dev_type = self.device.type

        dev_idx = (
            f',{self.device.index}'
            if (self.device.index is not None)
            else ''
        )

        dev_cname = (
            f' ({torch.cuda.get_device_name(self.device)})'
            if (dev_type == 'cuda')
            else ''
        )

        return f'{dev_type}{dev_idx}{dev_cname}'

    def _log_gpu_memory(self):
        if self.device.type == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(self.device)

            print('Maximum bytes of GPU memory used:')
            print(max_mem)

    def _save_opts(self, opt):
        opt_path = os.path.join(self.log_path, 'opt.json')

        with open(opt_path, 'w') as fd:
            json.dump(vars(opt), fd, indent=2)

    def _batch_to_device(self, batch_cpu):
        batch_gpu = list()

        for dataset_cpu in batch_cpu:
            dataset_gpu = dict()

            for k, ipt in dataset_cpu.items():
                if isinstance(ipt, torch.Tensor):
                    dataset_gpu[k] = ipt.to(self.device)

                else:
                    dataset_gpu[k] = ipt

            batch_gpu.append(dataset_gpu)

        return tuple(batch_gpu)

    def _validate_batch_segmentation(self, model, batch, score, images):
        if len(batch) != 1:
            raise Exception('Can only run validation on batches containing only one dataset')

        im_scores = list()
        single_im_score = RemappingScore()

        batch_gpu = self._batch_to_device(batch)
        outputs = model(batch_gpu)  # forward the data through the network

        colors_gt = batch[0]['color', 0, -1]
        segs_gt = batch[0]['segmentation', 0, 0].squeeze(1).long()
        segs_pred = outputs[0]['segmentation_logits', 0]
        segs_pred = functional.interpolate(segs_pred, segs_gt[0, :, :].shape, mode='nearest')

        for i in range(segs_pred.shape[0]):
            color_gt = colors_gt[i].unsqueeze(0)
            seg_gt = segs_gt[i].unsqueeze(0)
            seg_pred = segs_pred[i].unsqueeze(0)

            images.append((color_gt, seg_gt, seg_pred.argmax(1).cpu()))

            score.update(seg_gt, seg_pred)
            single_im_score.update(seg_gt, seg_pred)
            im_scores.append(single_im_score['none'].get_scores())
            single_im_score.reset()

        return im_scores

    def _run_segmentation_validation(self, images_to_keep=0, class_remaps=('none',)):
        scores = dict()
        images = dict()

        # torch.no_grad() = disable gradient calculation
        with torch.no_grad(), self.state.model_manager.get_eval() as model:
            for batch in self.segmentation_validation_loader:
                domain = batch[0]['domain'][0]

                if domain not in scores:
                    scores[domain] = RemappingScore(class_remaps)
                    images[domain] = list()

                _ = self._validate_batch_segmentation(model, batch, scores[domain], images[domain])

                images[domain] = images[domain][:images_to_keep]

        return scores, images