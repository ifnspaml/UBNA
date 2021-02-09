import torch
import torch.nn as nn
import torchvision.models as models

RESNETS = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}


class ResnetEncoder(nn.Module):
    """A ResNet that handles multiple input images and outputs skip connections"""

    def __init__(self, num_layers, pretrained):
        super().__init__()

        if num_layers not in RESNETS:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = RESNETS[num_layers](pretrained)

        # Remove fully connected layer
        self.encoder.fc = None

        if num_layers > 34:
            self.num_ch_enc = (64, 256,  512, 1024, 2048)
        else:
            self.num_ch_enc = (64, 64, 128, 256, 512)

    def forward(self, l_0):
        l_0 = self.encoder.conv1(l_0)
        l_0 = self.encoder.bn1(l_0)
        l_0 = self.encoder.relu(l_0)

        l_1 = self.encoder.maxpool(l_0)
        l_1 = self.encoder.layer1(l_1)

        l_2 = self.encoder.layer2(l_1)
        l_3 = self.encoder.layer3(l_2)
        l_4 = self.encoder.layer4(l_3)

        return (l_0, l_1, l_2, l_3, l_4)
