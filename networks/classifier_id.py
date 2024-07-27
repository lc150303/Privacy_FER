import torch.nn as nn
import numpy as np
from .networks import NetworkBase

class classifier_id(NetworkBase):
    """identity Classifier"""
    def __init__(self, opt):
        super(classifier_id, self).__init__(opt)
        self._name = 'classifier_id'

        layers = []
        layers.append(nn.Linear(opt.feature_size//2,1024))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(1024,1024))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(1024,512))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(512,self._opt.subject_type))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        pose_logit = self.main(x)
        return pose_logit

        
