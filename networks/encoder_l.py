import torch.nn as nn
import numpy as np
from torchvision.models import resnet34
from .networks import NetworkBase, ResidualBlock
import torch


class ResNetEncoder_l(NetworkBase):
    """ out_dim should be En_h's out_dim/2 """
    def __init__(self, opt, out_dim):
        super(ResNetEncoder_l, self).__init__(opt)
        self._name = 'ResEn_l'

        self.main = resnet34()
        self.main.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        # print(self.main)

    def forward(self, x):
        feature_e = self.main(x)
        feature_map_e = feature_e.unsqueeze(-1).unsqueeze(-1)
        return feature_e, feature_map_e

class Encoder_l(NetworkBase):
    def __init__(self, opt, conv_dim=32, repeat_num=3):
        super(Encoder_l, self).__init__(opt)
        self._name = 'En_l'

        # Input layer
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(self._get_output_function(True))

        # for i in range(3):
        #     layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        #     layers.append(self._get_output_function(True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(self._get_output_function(True))
            curr_dim *= 2

        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers.append(self._get_output_function(True))

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        feature_map_e = self.main(x)
        feature_e = feature_map_e.view(feature_map_e.size(0), -1)
        return feature_e, feature_map_e


