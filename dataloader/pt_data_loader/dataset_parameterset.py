import os
import sys
import json
import numpy as np

import dataloader.file_io.get_path as gp
import dataloader.definitions.labels_file as lf


class DatasetParameterset:
    """A class that contains all dataset-specific parameters

        - K: Extrinsic camera matrix as a Numpy array. If not available, take None
        - stereo_T: Distance between the two cameras (see e.g. http://www.cvlibs.net/datasets/kitti/setup.php, 0.54m)
        - labels: Segmentation labels
        - labels_mode: 'fromid' or 'fromrgb', depending on which format the segmentation images have
        - depth_mode: 'uint_16' or 'uint_16_subtract_one' depending on which format the depth images have
        - flow_mode: specifies how the flow images are stored, e.g. 'kitti'
        - splits: List of splits that are available for this dataset
    """
    def __init__(self, dataset):
        path_getter = gp.GetPath()
        dataset_folder = path_getter.get_data_path()
        path = os.path.join(dataset_folder, dataset, 'parameters.json')
        if not os.path.isdir(os.path.join(dataset_folder, dataset)):
            raise Exception('There is no dataset folder called {}'.format(dataset))
        if not os.path.isfile(path):
            raise Exception('There is no parameters.json file in the dataset folder. Please make sure '
                            'to place the downloaded file into tha dataset folder.')
        with open(path) as file:
            param_dict = json.load(file)
        self._dataset = dataset
        self._K = param_dict['K']
        if self._K is not None:
            self._K = np.array(self._K, dtype=np.float32)
        if param_dict['stereo_T'] is not None:
            self._stereo_T = np.eye(4, dtype=np.float32)
            self._stereo_T[0, 3] = param_dict['stereo_T']
        else:
            self._stereo_T = None
        self._depth_mode = param_dict['depth_mode']
        self._flow_mode = param_dict['flow_mode']
        self._splits = param_dict['splits']
        labels_name = param_dict['labels']
        if labels_name in lf.dataset_labels.keys():
            self.labels = lf.dataset_labels[labels_name].getlabels()
            self.labels_mode = param_dict['labels_mode']
        else:
            self.labels = None
            self.labels_mode = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def K(self):
        return self._K

    @property
    def stereo_T(self):
        return self._stereo_T

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def flow_mode(self):
        return self._flow_mode

    @property
    def splits(self):
        return self._splits



