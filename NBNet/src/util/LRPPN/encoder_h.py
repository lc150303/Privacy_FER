import torch.nn as nn
import numpy as np
from torchvision.models import resnet34
if __name__ == '__main__':
    from networks import NetworkBase, ResidualBlock
else:
    from .networks import NetworkBase, ResidualBlock
import torch

class ResNetEncoder_h(NetworkBase):
    def __init__(self, opt, out_dim):
        super(ResNetEncoder_h, self).__init__(opt)
        self._name = 'ResEn_h'

        self.main = resnet34()
        self.main.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        # print(self.main)

        self.output_size = 512
        self.output_dim = 512


    def forward(self, x):
        feat = self.main(x)
        feature_id, feature_e = feat.split(feat.size(1)//2, 1)
        _map = feat.unsqueeze(-1).unsqueeze(-1)
        feature_map_id, feature_map_e = _map.split(_map.size(1)//2, 1)
        return feature_id, feature_e, feature_map_id, feature_map_e


class Encoder_h(NetworkBase):
    def __init__(self, opt, conv_dim=64, repeat_num=3):
        super(Encoder_h, self).__init__(opt)
        self._name = 'En_h'

        # Input layer
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(self._get_output_function(True))

        # for i in range(3):
        #     layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        #     layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        #     layers.append(self._get_output_function(True))

        # Down-Sampling height and weight //= 2^4
        curr_dim = conv_dim
        cur_size = opt.HR_image_size
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(self._get_output_function(True))
            curr_dim *= 2
            cur_size //= 2

        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers.append(self._get_output_function(True))
        cur_size //= 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # output
        cur_size = cur_size**2*curr_dim
        # layers.append(nn.Linear(cur_size, cur_size//2))
        # layers.append(nn.Linear(cur_size//2, opt.feature_size))

        self.main = nn.Sequential(*layers)
        self.output_size = cur_size
        self.output_dim = curr_dim

    def forward(self, x):
        _map = self.main(x)
        feature_map_id, feature_map_e = _map.split(_map.size(1)//2, 1)
        feature_id, feature_e = feature_map_id.view(feature_map_id.size(0), -1), feature_map_e.view(feature_map_e.size(0), -1)
        return feature_id, feature_e, feature_map_id, feature_map_e
        


if __name__ == '__main__':
    rse = ResNetEncoder(1, 1)