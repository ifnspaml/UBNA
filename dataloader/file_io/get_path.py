import os
import platform
import socket
import json

class GetPath:
    def __init__(self):
        """This class gives the paths that are needed for training and testing neural networks.

        Paths that need to be specified are the data path and the checkpoint path, where the models will be saved.
        The paths have to saved in environment variables called UBNA_DIR_DATASET and UBNA_DIR_CHECKPOINT, respectively.
        """

        # Check if the user did explicitly set environment variables
        if self._guess_by_env():
            return

        # Print a helpful text when no directories could be found
        if platform.system() == 'Windows':
            raise ValueError(
                'Could not determine dataset/checkpoint directory. '
                'You can use environment variables to specify these directories '
                'by using the following commands:\n'
                'setx UBNA_DIR_DATASET <PATH_TO_DATASET>\n'
                'setx UBNA_DIR_CHECKPOINT <PATH_TO_CHECKPOINTS>\n'
            )

        else:
            raise ValueError(
                'Could not determine dataset/checkpoint directory. '
                'You can use environment variables to specify these directories '
                'by adding lines like the  following to your ~/.bashrc:\n'
                'export UBNA_DIR_DATASET=<PATH_TO_DATASET>\n'
                'export UBNA_DIR_CHECKPOINT=<PATH_TO_CHECKPOINTS>'
            )

    def _check_dirs(self):
        if self.dataset_base_path is None:
            return False

        if self.checkpoint_base_path is None:
            return False

        if not os.path.isdir(self.dataset_base_path):
            return False

        return True

    def _guess_by_env(self):
        dataset_base = os.environ.get('UBNA_DIR_DATASET', None)
        checkpoint_base = os.environ.get('UBNA_DIR_CHECKPOINT', None)

        self.dataset_base_path = dataset_base
        self.checkpoint_base_path = checkpoint_base

        return self._check_dirs()

    def get_data_path(self):
        """returns the path to the dataset folder"""

        return self.dataset_base_path

    def get_checkpoint_path(self):
        """returns the path to the checkpoints of the models"""

        return self.checkpoint_base_path
