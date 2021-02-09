from torch.utils.data import DataLoader, ConcatDataset

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as tf


def kitti_kitti_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads images for adaptation from the kitti training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height, resize_width),
            image_types=('color',)
        ),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'kitti_kitti_adaptation'),
        tf.AddKeyValue('purposes', ('adaptation',)),
    ]

    dataset_name = 'kitti'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'mono',
        'stereo_mode': 'mono',
        'split': 'kitti_split'
    }

    cfg_left = {'keys_to_load': ('color',),
                'keys_to_video': ('color',)}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the kitti (kitti_split) train set for adaptation", flush=True)

    return loader


def cityscapes_sequence_train(resize_height, resize_width, crop_height, crop_width, batch_size, num_workers):
    """A loader that loads images for adaptation from the cityscapes_sequence training set.
    This loader returns sequences from the left camera, as well as from the right camera.
    """

    transforms_common = [
        tf.RandomHorizontalFlip(),
        tf.CreateScaledImage(),
        tf.Resize(
            (resize_height * 568 // 512, resize_width * 1092 // 1024),
            image_types=('color',)
        ),
        # crop away the sides and bottom parts of the image
        tf.SidesCrop(
            (resize_height * 320 // 512, resize_width * 1024 // 1024),
            (resize_height * 32 // 512, resize_width * 33 // 1024)
        ),
        tf.CreateColoraug(new_element=True),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, gamma=0.0),
        tf.RemoveOriginals(),
        tf.ToTensor(),
        tf.NormalizeZeroMean(),
        tf.AddKeyValue('domain', 'cityscapes_sequence_adaptation'),
        tf.AddKeyValue('purposes', ('adaptation',)),
    ]
    
    dataset_name = 'cityscapes_sequence'

    cfg_common = {
        'dataset': dataset_name,
        'trainvaltest_split': 'train',
        'video_mode': 'mono',
        'stereo_mode': 'mono',
    }

    cfg_left = {'keys_to_load': ('color',),
                'keys_to_video': ('color',)}

    cfg_right = {'keys_to_load': ('color_right',),
                 'keys_to_video': ('color_right',)}

    dataset_left = StandardDataset(
        data_transforms=transforms_common,
        **cfg_left,
        **cfg_common
    )

    dataset_right = StandardDataset(
        data_transforms=[tf.ExchangeStereo()] + transforms_common,
        **cfg_right,
        **cfg_common
    )

    dataset = ConcatDataset((dataset_left, dataset_right))

    loader = DataLoader(
        dataset, batch_size, True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"  - Can use {len(dataset)} images from the cityscapes_sequence train set for adaptation", flush=True)

    return loader
