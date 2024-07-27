import torch.nn as nn
import functools

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
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        nn.init.kaiming_normal_(m.weight.data, a=0.02)

def weight_init_xavier_noraml(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight.data)

def weight_init_xavier_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_uniform_(m.weight.data)

if __name__ == '__main__':
    pass