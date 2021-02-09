#!/usr/bin/env python3

# Python standard library
import os

# Public libraries
import torch
import tensorboardX as tensorboard

# Local imports
import colors
import loaders, loaders.segmentation

from arguments import TrainingArguments
from timer import Timer
from harness import Harness
from losses import SegLosses


class Trainer(Harness):
    def _init_losses(self, opt):
        self.seg_losses = SegLosses(self.device)

    def _init_logging(self, opt):
        self.print_frequency = opt.train_print_frequency
        self.tb_frequency = opt.train_tb_frequency
        self.checkpoint_frequency = opt.train_checkpoint_frequency

    def _init_tensorboard(self, opt):
        self.writers = dict(
            (mode, tensorboard.SummaryWriter(os.path.join(self.log_path, mode)))
            for mode in ('train', 'validation', 'images')
        )

    def _init_train_loaders(self, opt):
        print('Loading training dataset metadata:', flush=True)

        # Make sure that either the model is trained or adapted
        if opt.segmentation_training_loaders == '':
            raise Exception('The segmanetation_training_loaders have to be non-empty '
                            'to perform the training')

        # Directly call the loader setup functions from loaders/adaptation and loaders/segmentation
        # that are passed in via --loaders_adaptation and --loaders_segmentation.

        segmentation_train_loaders = list(
            getattr(loaders.segmentation, loader_name)(
                resize_height=opt.segmentation_resize_height,
                resize_width=opt.segmentation_resize_width,
                crop_height=opt.segmentation_crop_height,
                crop_width=opt.segmentation_crop_width,
                batch_size=opt.segmentation_training_batch_size,
                num_workers=opt.sys_num_workers
            )
            for loader_name in opt.segmentation_training_loaders.split(',') if (loader_name != '')
        )

        self.train_loaders = loaders.FixedLengthLoaderList(
            segmentation_train_loaders,
            opt.train_batches_per_epoch
        )

    def _init_training(self, opt):
        self.num_epochs = opt.train_num_epochs

    def _flush_logging(self):
        print('', end='', flush=True)

        for writer in self.writers.values():
            writer.flush()

    def _log_seg(self, domain_name, batch_idx, inputs, outputs, losses):
        with torch.no_grad():
            # Multiple times each epoch ...
            if (batch_idx % self.tb_frequency) == 0:
                # ... log the segmentation loss to tensorboard
                self.writers['train'].add_scalar(
                    f"{domain_name}_loss", losses["loss_seg"].cpu(), self.state.step
                )

            # A few times each epoch ...
            if (batch_idx % self.print_frequency) == 0:
                print(f"  - {domain_name} losses at epoch {self.state.epoch} (batch {batch_idx}):")

                # ... log the cross entropy loss
                loss_seg = losses["loss_seg"].cpu()
                print(f"    - cross_entropy: {loss_seg:.4f}")

            # Once at the start of each epoch ...
            if batch_idx == 0:
                seg = outputs['segmentation_logits', 0].softmax(1).cpu()
                gt = inputs['segmentation', 0, 0][:, 0, :, :].cpu().long()
                src = inputs['color', 0, 0].cpu()

                logged_images = (
                    colors.seg_prob_image(seg),
                    colors.seg_idx_image(gt),
                    src
                )

                self.writers['images'].add_images(
                    f"{domain_name}_images",
                    torch.cat(logged_images, 2),
                    self.state.step
                )

    def _process_batch_seg(self, dataset, output, batch_idx, domain_name):
        if ('segmentation_logits', 0) not in output:
            return 0

        losses_seg = self.seg_losses.seg_losses(dataset, output)

        self._log_seg(domain_name, batch_idx, dataset, output, losses_seg)

        return losses_seg["loss_seg"]

    def _run_epoch(self):
        print(f"Epoch {self.state.epoch}:")

        with self.state.model_manager.get_train() as model:
            timer = Timer()

            timer.enter('loading')
            for batch_idx, batch in enumerate(self.train_loaders):

                timer.enter(f"optimizer")
                self.state.optimizer.zero_grad()

                timer.enter(f"transfer")
                batch = self._batch_to_device(batch)

                timer.enter('forward')
                outputs = model(batch)

                loss_seg = 0

                for dataset, output in zip(batch, outputs):
                    domain_name = dataset['domain'][0]

                    # Calculate loss for the segmentation prediction. If no gt segmentation is available,
                    # then the loss will not be updated
                    loss_seg += self._process_batch_seg(dataset, output, batch_idx, domain_name)

                timer.enter(f"optimizer")

                # Compute the loss and the gradients, log data to tensorboard.
                loss_seg.backward()

                # Update the network parameters. During adaptation (as there are no gradients)
                # only the BN statistics are updated
                self.state.optimizer.step()

                if (batch_idx % self.print_frequency) == 0:
                    print('  - Breakdown of time spent this epoch:')
                    for category, t in timer.items():
                        print(f'    - {category}: {t:.3f}', flush=True)

                self.state.step += 1

                timer.enter('loading')

        self.state.lr_scheduler.step()

    def _run_validation(self):
        print(f'Validation scores for epoch {self.state.epoch}:')

        segmentation_scores, _ = self._run_segmentation_validation()

        for domain, score in segmentation_scores.items():
            metrics = score['none'].get_scores()

            print(f'  - {domain}:')

            for metric in sorted(metrics):
                value = metrics[metric]

                if metric in ('iou', 'acc', 'prec'):
                    # ignore non-scalars
                    continue

                print(f'    - {metric}: {value:.4f}')

                self.writers['validation'].add_scalar(
                    f"{domain}_{metric}", value, self.state.step
                )

    def train(self):
        while self.state.epoch < self.num_epochs:
            self._run_epoch()
            self._run_validation()
            self._flush_logging()

            self.state.epoch += 1

            # Save after save frequency
            if (self.state.epoch % self.checkpoint_frequency) == 0:
                self.state.store_checkpoint()

        # Save at end of training
        self.state.store_checkpoint()

        print('Completed without errors', flush=True)
        self._log_gpu_memory()


if __name__ == "__main__":
    opt = TrainingArguments().parse()
    if opt.sys_best_effort_determinism:
        import numpy as np
        import random

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    trainer = Trainer(opt)
    trainer.train()
