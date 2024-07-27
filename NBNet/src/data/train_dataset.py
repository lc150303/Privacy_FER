import os.path
from src.data.dataset import DatasetBase
from PIL import Image
import cv2
import random


class TrainDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(TrainDataset, self).__init__(opt, is_for_train)
        self._name = 'TrainDataset'
        self.is_pretrain = None
        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        
        # get sample data
        HR_id, beha, subj, LR_id = self._data[index]

        HR_img_1, HR_img_path_1 = self._get_img_by_id(HR_id)
        HR_img_1 = cv2.resize(HR_img_1, (self._opt.HR_image_size, self._opt.HR_image_size))
        LR_img_1, LR_img_path_1 = self._get_img_by_id(LR_id)
        LR_img_1 = cv2.resize(LR_img_1, (self._opt.HR_image_size, self._opt.HR_image_size))
        # transform data
        HR_img_1 = self._transform(Image.fromarray(HR_img_1))
        LR_img_1 = self._transform(Image.fromarray(LR_img_1))

        if self.is_pretrain:
            return {
                'HR_img': HR_img_1,
                'HR_img_path': HR_img_path_1,
                'HR_beha': beha,
                'HR_subj': subj,
                'LR_img': LR_img_1,
                'LR_img_path': LR_img_path_1,
                'LR_beha': beha,
                'LR_subj': subj,
            }

        HR_id_2, _, subj_2, LR_id_2 = self.get_pair_sample(beha, HR_id)

        HR_img_2, HR_img_path_2 = self._get_img_by_id(HR_id_2)
        HR_img_2 = cv2.resize(HR_img_2, (self._opt.HR_image_size, self._opt.HR_image_size))
        LR_img_2, LR_img_path_2 = self._get_img_by_id(LR_id_2)
        LR_img_2 = cv2.resize(LR_img_2, (self._opt.HR_image_size, self._opt.HR_image_size))
        # transform data
        HR_img_2 = self._transform(Image.fromarray(HR_img_2))
        LR_img_2 = self._transform(Image.fromarray(LR_img_2))

        #print (HR_img_1.shape)
        
        # pack data
        return {
            'HR_img_1': HR_img_1,
            'HR_img_path_1': HR_img_path_1,
            'LR_img_1': LR_img_1,
            'LR_img_path_1': LR_img_path_1,
            'beha_1': beha,
            'subj_1': subj,
            'HR_img_2': HR_img_2,
            'HR_img_path_2': HR_img_path_2,
            'LR_img_2': LR_img_2,
            'LR_img_path_2': LR_img_path_2,
            'beha_2': beha,
            'subj_2': subj_2
        }

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):

        # read ids
        ids_filename = 'train_ids' + self._opt.ids_file_suffix #if self._is_for_train else self._opt.HR_test_ids_file
        #target_ids_filename = self._opt.target_train_ids_file if self._is_for_train else self._opt.target_test_ids_file

        ids_filepath = os.path.join(self._root, ids_filename)
        #target_ids_filepath = os.path.join(self._root, target_ids_filename)
        
        _HR_ids, _behas, _subj, _LR_ids = self._read_ids(ids_filepath)
        _behas = [int(i) for i in _behas]
        _subj = [int(i) for i in _subj]

        self._data = list(zip(_HR_ids, _behas, _subj, _LR_ids))
        # dataset size
        self._dataset_size = len(self._data)

        # group by behas
        self._behas = self._group_by_behas(self._data)
        # for b in self._behas:
        #     print(len(self._behas[b]), self._behas[b])
        # assert 1==0

    def _read_ids(self, file_path):
        # self._HR_ids, self._HR_behas, self._HR_subjs, self._LR_ids, self._LR_behas, self._LR_subjs
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

    def _group_by_behas(self, data):
        behas = {}
        for d in data:
            if d[1] in behas:
                behas[d[1]].append(d)
            else:
                behas[d[1]] = [d]

        return behas

    def set_pretrain(self):
        self.is_pretrain = True

    def set_tune(self):
        self.is_pretrain = False

    def get_pair_sample(self, beha, HR_id):
        sample = random.sample(self._behas[beha], 1)[0]
        # print(sample)
        while sample[0] == HR_id:
            sample = random.sample(self._behas[beha], 1)[0]
        return sample

if __name__ == '__main__':
    dataset = TrainDataset({'data_dir'})