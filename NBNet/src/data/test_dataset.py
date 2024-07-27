import os.path

import numpy as np
from data.dataset import DatasetBase
from PIL import Image
import cv2


class TestDataset(DatasetBase):
    def __init__(self, opt):
        super(TestDataset, self).__init__(opt)
        self._name = 'Low Resolution TestDataset'
        
        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        
        # get sample data
        LR_img, LR_img_path = self._get_img_by_id(self._LR_ids[index])
        # print('readed img', type(LR_img), LR_img.shape if isinstance(LR_img, np.ndarray) else LR_img)
        LR_img = cv2.resize(LR_img, (self._opt.HR_image_size, self._opt.HR_image_size))

        HR_img, HR_img_path = self._get_img_by_id(self._HR_ids[index])
        HR_img = cv2.resize(HR_img, (self._opt.HR_image_size, self._opt.HR_image_size))

        LR_beha = self._LR_behas[index]
        LR_subj = self._LR_subjs[index]

        # # transform data
        # LR_img = self._transform(Image.fromarray(LR_img))
        # HR_img = self._transform(Image.fromarray(HR_img))
        
        # pack data
        sample = {'LR_img': LR_img,
                  'LR_img_path': LR_img_path,
                  'HR_img': HR_img,
                  'HR_img_path': HR_img_path,
                  'LR_beha': LR_beha,
                  'LR_subj': LR_subj
                      }
        
        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):

        # read ids
        ids_filename = 'test_ids' + self._opt.ids_file_suffix
        ids_filepath = os.path.join(self._root, ids_filename)
        self._HR_ids, self._LR_behas, self._LR_subjs, self._LR_ids = self._read_ids(ids_filepath)
        self._LR_behas = [int(i) for i in self._LR_behas]
        self._LR_subjs = [int(i) for i in self._LR_subjs]

        # dataset size
        self._dataset_size = len(self._LR_ids)

    def _read_ids(self, file_path):
        ids_and_labels = [[], [], [], []]
        with open(file_path, 'r') as f:
            for line in f:
                if line == '' or line == '\n':
                    continue
                labels = line.strip("\n").split(sep=',')
                assert len(labels) == 4, 'label file incomplete'
                for i in range(len(labels)):
                    ids_and_labels[i].append(labels[i])

        return ids_and_labels


