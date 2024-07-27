#!/usr/bin/python3
# coding=utf-8
"""
@Author: 梁聪
@last modified: 2022/5/1 15:03
"""
import os
import torch
import torchvision.transforms as transforms
import cv2
from LRPPN.encoder_l import Encoder_l
from LRPPN.encoder_h import Encoder_h
import numpy as np
from PIL import Image


class LRPPN():
    def __init__(self, opt, is_HR, model_path:str, device_id:int, img_size=128):
        """
        :param device_id: 单一数字，指定 GPU
        :param img_size: int，图片边长，输入格式为 (3, origin_size, origin_size)，会被本模型放缩到 (3, img_size, img_size)
        """
        self._is_HR = is_HR
        self._img_size = img_size or 16
        print('********** LRPPN img size', self._img_size)
        self._device = torch.device('cuda', device_id)

        # init structure, load weights, load onto GPU
        if is_HR:
            self.networks = {'En': Encoder_h(opt)}
        else:
            self.networks = {'En': Encoder_l(opt)}
        self._load_network(model_path, self._device)
        self.networks['En'].to(self._device).eval()

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5]),
                          ]
        self._transform = transforms.Compose(transform_list)

    def _load_network(self, load_path, device):
        assert os.path.exists(
            load_path), '%s Weights file not found. Have you trained a model!? We are not providing one' % load_path

        self.networks['En'].load_state_dict(torch.load(load_path, map_location=device))
        print ('LRPPN loaded net: %s' % load_path)

    def get_feature(self, images:np.ndarray):
        """
        :param images: (batch_size, 3, origin_size, origin_size)
        """
        # print('LPPRN.get_feature(): input type ', type(images))
        # print('LPPRN.get_feature(): input shape ', images.shape, images.dtype)
        tensor_imgs = torch.stack([self._transform(cv2.resize(img, (self._img_size, self._img_size))) for img in images],  dim=0).to(self._device)
        # print('LRPPN input shape', tensor_imgs.shape)

        with torch.no_grad():
            if self._is_HR:
                _, features, _, _ = self.networks['En'](tensor_imgs)
            else:
                features, _ = self.networks['En'](tensor_imgs)
        # print('LRPPN out shape', features.shape)

        return features.cpu()