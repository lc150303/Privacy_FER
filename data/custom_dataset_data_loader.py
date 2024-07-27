import torch.utils.data
from data.dataset import DatasetFactory

class CustomDatasetDataLoader:
    def __init__(self, opt, dataset_mode, is_for_train=True):
        self._opt = opt
        self._is_for_train = is_for_train
        self._num_threds = opt.n_threads_train if is_for_train else opt.n_threads_test
        self._dataset_mode = dataset_mode
        self._create_dataset()

    def _create_dataset(self):
        self._dataset = DatasetFactory.get_by_name(self._dataset_mode, self._opt, self._is_for_train)
        if hasattr(self._dataset, 'feature_size'):
            self.feature_size = self._dataset.feature_size
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self._opt.batch_size//2 if self._is_for_train else 1,
            shuffle=self._is_for_train,
            num_workers=int(self._num_threds),
            drop_last=self._is_for_train,
            pin_memory=False)

    def load_data(self):
        return self._dataloader

    def __len__(self):
        return len(self._dataset)

    def set_pretrain(self):
        self._dataset.set_pretrain()

    def set_tune(self):
        self._dataset.set_tune()