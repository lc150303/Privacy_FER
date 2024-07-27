import torch.nn as nn
import numpy as np
from .networks import NetworkBase

class discriminator_id_r(NetworkBase):
    """subject&reality Discriminator"""
    def __init__(self, opt, conv_dim=64, repeat_num=2):
        super(discriminator_id_r, self).__init__(opt)
        self._name = 'discriminator_id_r'

        # Input layer
        conv_layers = []
        conv_layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        conv_layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        conv_layers.append(self._get_output_function(False))

        # Downsample layer
        cur_dim = conv_dim
        cur_size = opt.HR_image_size
        for i in range(repeat_num):
            conv_layers.append(nn.Conv2d(cur_dim, cur_dim // 4, kernel_size=4, stride=2, padding=1, bias=False))
            conv_layers.append(nn.InstanceNorm2d(cur_dim // 4, affine=True))
            conv_layers.append(self._get_output_function(False))
            cur_dim = cur_dim // 4
            cur_size = cur_size // 2

        linear_layers = []
        linear_layers.append(nn.Linear(cur_size ** 2 * cur_dim, 1024))
        linear_layers.append(nn.Linear(1024, 512))
        linear_layers.append(nn.Linear(512, opt.subject_type+1))

        self.conv = nn.Sequential(*conv_layers)
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        feature_maps = self.conv(x)
        subject_r_logit = self.linear(feature_maps.view(feature_maps.size(0), -1))
        return subject_r_logit.split(subject_r_logit.shape[1]-1, 1)

        
