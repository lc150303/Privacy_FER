import traceback

import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):
        opt = args[0]

        if network_name == 'En_h':
            from .encoder_h import Encoder_h, ResNetEncoder_h
            if opt.resnet:
                network = ResNetEncoder_h(opt, 512)
            else:
                network = Encoder_h(*args, **kwargs)
        elif network_name == 'En_l':
            from .encoder_l import Encoder_l, ResNetEncoder_l
            if opt.resnet:
                network = ResNetEncoder_l(opt, 256)
            else:
                network = Encoder_l(*args, **kwargs)
        elif network_name == 'C_e':
            from .classifier_e import classifier_e
            network = classifier_e(*args, **kwargs)
        elif network_name == 'C_e_adv':
            from .classifier_e import classifier_e
            network = classifier_e(*args, **kwargs)
        elif network_name == 'C_id':
            from .classifier_id import classifier_id
            network = classifier_id(*args, **kwargs)
        elif network_name == 'C_id_adv':
            from .classifier_id import classifier_id
            network = classifier_id(*args, **kwargs)
        elif network_name == 'Dis_e':
            from .discriminator_e import discriminator_e
            network = discriminator_e(*args, **kwargs)
        elif network_name == 'Dis_id_r':
            from .discriminator_id_r import discriminator_id_r
            network = discriminator_id_r(*args, **kwargs)
        elif network_name == 'De':
            from .decoder import decoder, ResNetDecoder
            if opt.resnet:
                network = ResNetDecoder(opt, 512)
            else:
                network = decoder(*args, **kwargs)
        elif network_name == 'C_l':
            from .classifier_l import classifier_LR
            network = classifier_LR(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print ("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self, opt):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'
        self._opt = opt

    @property
    def name(self):
        return self._name

    def init_weights(self, method:str):
        if method == 'normal':
            self.apply(weights_init_normal)
        elif method == 'kaiming_normal':
            self.apply(weight_init_kaiming_normal)
        elif method == 'kaiming_uniform':
            self.apply(weight_init_kaiming_uniform)
        elif method == 'xavier_normal':
            self.apply(weight_init_xavier_noraml)
        elif method == 'xavier_uniform':
            self.apply(weight_init_xavier_uniform)
        else:
            raise NotImplementedError('%s weight initialization not recognized')


    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer

    def _get_output_function(self, inplace):
        return nn.LeakyReLU(negative_slope=0.02, inplace=inplace)


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


def weights_init_normal(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weight_init_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        nn.init.kaiming_uniform_(m.weight.data, a=0.02)

def weight_init_kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        try:
            nn.init.kaiming_normal_(m.weight.data, a=0.02)
        except:
            print(m)
            traceback.print_exc()
            raise Exception()

def weight_init_xavier_noraml(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight.data)

def weight_init_xavier_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_uniform_(m.weight.data)

if __name__ == '__main__':
    a = new