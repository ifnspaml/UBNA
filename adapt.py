#!/usr/bin/env python3

# Public libraries
import torch

# Local imports
import loaders, loaders.adaptation

from harness import Harness
from batchwise_adaptation import BatchWiseAdaptation
from arguments import AdaptationArguments


class Trainer(Harness):
    def _init_losses(self, opt):
        pass

    def _init_logging(self, opt):
        self.print_frequency = opt.train_print_frequency
        self.tb_frequency = opt.train_tb_frequency
        self.checkpoint_frequency = opt.train_checkpoint_frequency

    def _init_tensorboard(self, opt):
        pass

    def _init_train_loaders(self, opt):
        print('Loading Adaptation dataset metadata:', flush=True)

        # Make sure that only the adaptation loader contains entries
        if opt.adaptation_training_loaders == '':
            raise Exception('The adaptation_training_loaders needs to contain entries')

        # Directly call the loader setup functions from loaders/adaptation and loaders/segmentation
        # that are passed in via --loaders_adaptation and --loaders_segmentation.

        adaptation_loaders = list(
            getattr(loaders.adaptation, loader_name)(
                resize_height=opt.adaptation_resize_height,
                resize_width=opt.adaptation_resize_width,
                crop_height=opt.adaptation_crop_height,
                crop_width=opt.adaptation_crop_width,
                batch_size=opt.adaptation_training_batch_size,
                num_workers=opt.sys_num_workers
            )
            for loader_name in opt.adaptation_training_loaders.split(',') if (loader_name != '')
        )

        self.adaptation_loaders = loaders.FixedLengthLoaderList(
            adaptation_loaders,
            opt.train_batches_per_epoch
        )

    def _init_training(self, opt):
        self.num_batches = opt.adaptation_num_batches
        self.xlsx_frequency = opt.adaptation_xlsx_frequency
        self.eval_remaps = opt.segmentation_eval_remaps.split(',')
        self.sequential_batch_training = BatchWiseAdaptation(opt.adaptation_mode_sequential,
                                                             opt.adaptation_training_batch_size,
                                                             opt.adaptation_batchnorm_momentum,
                                                             opt.adaptation_alpha_batch,
                                                             opt.adaptation_alpha_layer)

    def _flush_logging(self):
        print('', end='', flush=True)

        for writer in self.writers.values():
            writer.flush()

    def _run_adaptation(self):
        """performs the adaptation """

        print(f"Epoch {self.state.epoch}:")
        print(f"Adaptation initialized")

        if self.num_batches == 0:
            raise Exception('please specify the number of adaptation batches (i.e adaptation steps)!')

        with self.state.model_manager.get_train() as model:
            for batch_idx, batch in enumerate(self.adaptation_loaders):
                print(f"Performing adaptation step nr {batch_idx}!")

                # Check for the end of sequential training
                if batch_idx == self.num_batches:
                    directory_naming = f"batch_{batch_idx}"
                    self.state.store_batch_checkpoint(directory_naming)  # Store checkpoint at the end of training
                    print(f'adaptation of {self.num_batches} is completed')
                    break

                model = self.sequential_batch_training.process(model, batch_idx)

                # timer.enter(f"transfer")
                batch = self._batch_to_device(batch)

                # timer.enter('forward')
                outputs = model(batch)

    def adapt(self):
        self._run_adaptation()

        print('Completed adaptation without errors', flush=True)
        self._log_gpu_memory()


if __name__ == "__main__":
    opt = AdaptationArguments().parse()

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
    # trainer.train()
    trainer.adapt()
