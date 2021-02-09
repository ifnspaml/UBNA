from __future__ import absolute_import, division, print_function

from dataloader.pt_data_loader.basedataset import BaseDataset


class StandardDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(StandardDataset, self).__init__(*args, **kwargs)

        if self.disable_const_items is False:
            assert self.parameters.K is not None and self.parameters.stereo_T is not None, '''There are no K matrix and
            stereo_T parameter available for this dataset.'''

    def add_const_dataset_items(self, sample):
        K = self.parameters.K.copy()

        native_key = ('color', 0, -1) if (('color', 0, -1) in sample) else ('color_right', 0, -1)
        native_im_shape = sample[native_key].shape

        K[0, :] *= native_im_shape[1]
        K[1, :] *= native_im_shape[0]

        sample["K", -1] = K
        sample["stereo_T"] = self.parameters.stereo_T

        return sample