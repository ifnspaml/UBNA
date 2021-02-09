import torch
import torch.nn as nn

from . import networks


class UBNACommon(nn.Module):
    def __init__(self, num_layers, pretrained=False):
        super().__init__()
        self.encoder = networks.VggEncoder(num_layers, pretrained)
        self.num_layers = num_layers  # This information is needed in the train loop for the sequential training

        # Number of channels for the skip connections and internal connections
        # of the decoder network, ordered from input to output
        self.shape_enc = tuple(reversed(self.encoder.num_ch_enc))
        self.shape_dec = (256, 128, 64, 32, 16)

        split_pos = 1
        self.decoder = networks.PartialDecoder.gen_head(self.shape_dec, self.shape_enc, split_pos)

    def forward(self, x):
        # The encoder produces outputs in the order
        # (highest res, second highest res, …, lowest res)
        x = self.encoder(x)

        # The decoder expects it's inputs in the order they are
        # used. E.g. (lowest res, second lowest res, …, highest res)
        x = tuple(reversed(x))

        # Replace some elements in the x tuple by decoded
        # tensors and leave others as-is
        x = self.decoder(*x)  # CHANGE ME BACK TO THIS

        return x


class UBNASeg(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.decoder = networks.PartialDecoder.gen_tail(encoder.decoder)
        self.multires = networks.MultiResSegmentation(self.decoder.chs_x()[-1:])
        self.nl = nn.Softmax2d()

    def forward(self, *x):
        x = self.decoder(*x)
        x = self.multires(*x[-1:])
        x_lin = x[-1]

        return x_lin


class UBNAVGG(nn.Module):
    KEY_FRAME_CUR = ('color_aug', 0, 0)

    def __init__(self, num_layers=18, weights_init='pretrained'):
        super().__init__()

        if weights_init == 'pretrained':
            pretrained = True
        elif weights_init == 'scratch':
            pretrained = False
        else:
            raise Exception('Unsupported weights init mode')

        self.common = UBNACommon(
            num_layers, pretrained=pretrained
        )

        self.seg = UBNASeg(self.common)

    def _batch_pack(self, group):
        # Concatenate a list of tensors and remember how
        # to tear them apart again

        group = tuple(group)

        dims = tuple(b.shape[0] for b in group)
        group = torch.cat(group, dim=0)  # concatenate along the first axis, so along the batch axis

        return dims, group

    def _multi_batch_unpack(self, dims, *xs):
        xs = tuple(
            tuple(x.split(dims))
            for x in xs
        )

        # xs, as of now, is indexed like this:
        # xs[ENCODER_LAYER][DATASET_IDX], the lines below swap
        # this around to xs[DATASET_IDX][ENCODER_LAYER], so that
        # xs[DATASET_IDX] can be fed into the decoders.
        xs = tuple(zip(*xs))

        return xs

    def _check_purposes(self, dataset, purpose):
        # mytransforms.AddKeyValue is used in the loaders
        # to give each image a tuple of 'purposes'.
        # As of now these purposes can be 'adaptation' and 'segmentation'.
        # The torch DataLoader collates these per-image purposes
        # into list of them for each batch.
        # Check all purposes in this collated list for the requested
        # purpose (if you did not do anything wonky all purposes in a
        # batch should be equal),

        for purpose_field in dataset['purposes']:
            if purpose_field[0] == purpose:
                return True

    def forward(self, batch):
        # Stitch together all current input frames
        # in the input group. So that batch normalization
        # in the encoder is done over all datasets/domains.
        dims, x = self._batch_pack(
            dataset[self.KEY_FRAME_CUR]
            for dataset in batch
        )

        x_seg = self.common(x)

        # Cut the stitched-together tensors along the
        # dataset boundaries so further processing can
        # be performed on a per-dataset basis.
        # x[DATASET_IDX][ENCODER_LAYER]
        x_seg = self._multi_batch_unpack(dims, *x_seg)

        # Cut the stitched-together tensors along the
        # dataset boundaries so further processing could
        # be performed on a per-domain basis.
        # x[DATASET_IDX][ENCODER_LAYER]

        outputs = list(dict() for _ in batch)

        # All the way back in the loaders each dataset is assigned one or more 'purposes'.
        # which are then used here to decide whether a segmentation output is calculated
        for idx, dataset in enumerate(batch):
            if self._check_purposes(dataset, 'segmentation'):
                x = x_seg[idx]
                x = self.seg(*x)

                outputs[idx]['segmentation_logits', 0] = x

        return tuple(outputs)
