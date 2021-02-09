import torch

from .cityscapes import COLOR_SCHEME_CITYSCAPES


def seg_prob_image(probs):
    """Takes a torch tensor of shape (N, C, H, W) containing a map
    of cityscapes class probabilities C as input and generate a
    color image of shape (N, C, H, W) from it.
    """

    # Choose the number of categories according
    # to the dimesion of the input Tensor
    colors = COLOR_SCHEME_CITYSCAPES[:probs.shape[1]]

    # Make the category channel the last dimension (N, W, H, C),
    # matrix multiply so that the color channel is the last
    # dimension and restore the shape array to (N, C, H, W).
    image = (probs.transpose(1, -1) @ colors).transpose(-1, 1)

    return image


def seg_idx_image(idxs):
    """Takes a torch tensor of shape (N, H, W) containing a map
    of cityscapes train ids as input and generate a color image
    of shape (N, C, H, W) from it.
    """

    # Take the dimensionality from (N, H, W) to (N, C, H, W)
    # and make the tensor invariant over the C dimension
    idxs = idxs.unsqueeze(1)
    idxs = idxs.expand(-1, 3, -1, -1)

    h, w = idxs.shape[2:]

    # Extend the dimesionality of the color scheme from
    # (IDX, C) to (IDX, C, H, W) and make it invariant over
    # the last two dimensions.
    color = COLOR_SCHEME_CITYSCAPES.unsqueeze(2).unsqueeze(3)
    color = color.expand(-1, -1, h, w)

    image = torch.gather(color, 0, idxs)

    return image
