import os
import sys
import wget
import zipfile
import pandas as pd
import shutil
import numpy as np
import glob
import cv2

import dataloader.file_io.get_path as gp
import dataloader.file_io.dir_lister as dl


def download_kitti_all(kitti_folder='kitti_download'):
    """ This python script downloads all KITTI folders and arranges them in a
    coherent data structure which can respectively be used by the other data
    scripts. It is recommended to keep the standard name kitti. Note that the
    path is determined automatically inside of file_io/get_path.py

    :param kitti_folder: Name of the folder in which the dataset should be downloaded
                    This is no path but just a name. the path is determined by
                    get_path.py
    """
    path_getter = gp.GetPath()
    dataset_folder_path = path_getter.get_data_path()
    assert os.path.isdir(dataset_folder_path), 'Path to dataset folder does not exist'

    kitti_path = os.path.join(dataset_folder_path, kitti_folder)
    kitti_raw_data = pd.read_csv('kitti_archives_to_download.txt',
                                 header=None, delimiter=' ')[0].values
    kitti_path_raw = os.path.join(kitti_path, 'Raw_data')
    if not os.path.isdir(kitti_path_raw):
        os.makedirs(kitti_path_raw)
    for url in kitti_raw_data:
        folder = os.path.split(url)[1]
        folder = os.path.join(kitti_path_raw, folder)
        folder = folder[:-4]
        wget.download(url, out=kitti_path_raw)
        unzipper = zipfile.ZipFile(folder + '.zip', 'r')
        unzipper.extractall(kitti_path_raw)
        unzipper.close()
        os.remove(folder + '.zip')


if __name__ == '__main__':
    kitti_folder = 'kitti_download'
    download_kitti_all(kitti_folder)