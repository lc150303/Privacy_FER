import torch.utils.data
from data.test_dataset import TestDataset

class LRPPNDataLoader:
    def __init__(self, opt):
        self._opt = opt
        self._create_dataset()

    def _create_dataset(self):
        self._dataset = TestDataset(self._opt)
        if hasattr(self._dataset, 'feature_size'):
            self.feature_size = self._dataset.feature_size
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True)

    def load_data(self):
        return self._dataloader

    def __len__(self):
        return len(self._dataset)

if __name__ == '__main__':
    class Opt:
        pass

    opt = Opt()
    opt.HR_image_size = 128
    opt.ids_file_suffix = '_16.csv'
    opt.data_dir = '/home/liangcong/dataset/FERG'

    print(len(LRPPNDataLoader(opt)))

