import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path
import cv2

class DatasetBase(data.Dataset):
    def __init__(self, opt):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._opt = opt
        self._create_transform()
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, 'imgs')

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        # 变 [-1, 1]
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5]),
                          ]
        self._transform = transforms.Compose(transform_list)

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id)
        # print('imread', filepath)
        return cv2.imread(filepath, cv2.IMREAD_COLOR), filepath