import torch
import torch.nn as nn
import numpy as np
if __name__ == '__main__':
    from networks import NetworkBase, ResidualBlock
else:
    from .networks import NetworkBase, ResidualBlock
import torch
# from torchstat import stat

class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, short_cut=False, upsample=True):
        super(UpsampleBlock, self).__init__()
        self.short_cut = short_cut

        conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        bn1 = nn.BatchNorm2d(in_dim)
        relu = nn.ReLU(inplace=True)

        conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        bn2 = nn.BatchNorm2d(out_dim)

        layers = [conv1, bn1, relu, conv2, bn2]

        if upsample:
            layers = [nn.Upsample(scale_factor=2)] + layers
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        if self.short_cut:
            return self.main(x) + x
        else:
            return self.main(x)

class ResNetDecoder(NetworkBase):
    def __init__(self, opt, in_dim):
        # super(ResNetDecoder, self).__init__()
        super(ResNetDecoder, self).__init__(opt)
        self._name = 'ResDe'

        self.layer1 = nn.Sequential(
            UpsampleBlock(in_dim, 512),
            UpsampleBlock(512, 512, upsample=False, short_cut=True),
            UpsampleBlock(512, 512),
            UpsampleBlock(512, 256, upsample=False)
        )

        self.layer2 = nn.Sequential(
            UpsampleBlock(256, 256),
            UpsampleBlock(256, 256, upsample=False, short_cut=True),
            UpsampleBlock(256, 256),
            UpsampleBlock(256, 128, upsample=False)
        )

        self.layer3 = nn.Sequential(
            UpsampleBlock(128, 128),
            UpsampleBlock(128, 128, upsample=False, short_cut=True),
            UpsampleBlock(128, 128),
            UpsampleBlock(128, 64, upsample=False)
        )

        self.layer4 = nn.Sequential(
            UpsampleBlock(64, 64),
            UpsampleBlock(64, 64, upsample=False, short_cut=True),
            UpsampleBlock(64, 32),
            UpsampleBlock(32, 3, upsample=False)
        )


    def forward(self, id_map, e_map):
        x = torch.cat((id_map, e_map), 1)
        # print('cat map shape', x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print('decoder out shape', x.shape)
        # raise Exception()
        return x

class decoder(NetworkBase):
    """decoder"""
    def __init__(self, opt, repeat_num=3):
        super(decoder, self).__init__(opt)
        self._name = 'De'
        self.flag = True

        layers = []
        # Input-Layer
        layers.append(nn.Conv2d(opt.feature_dim, opt.feature_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(opt.feature_dim, affine=True))
        layers.append(self._get_output_function(False))

        # Up-Sampling
        curr_dim = opt.feature_dim
        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers.append(self._get_output_function(False))
        
        # Up-Sampling
        for i in range(repeat_num):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(self._get_output_function(False))
            curr_dim //= 2

        # Output-Layer
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

        # stat(self.main, input_size=(512, 8, 8))

    def forward(self, id_map, e_map):
        if self.flag:
            #print (id_map.shape)
            #print (e_map.shape)
            self.flag = False
        f_concat = torch.cat((id_map, e_map), 1)
        return self.main(f_concat)


if __name__ == '__main__':
    data = torch.rand((2, 512, 1, 1))
    resd = ResNetDecoder(1, 512)
    y = resd(data)
    print(y.shape)