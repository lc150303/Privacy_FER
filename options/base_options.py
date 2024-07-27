import argparse
import os
import time
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('--data_dir', type=str, default='sample_dataset', help='path to dataset')

        self._parser.add_argument('--ids_file_suffix', type=str, default='_ids3.csv', help='suffix of train/test ids file')

        self._parser.add_argument('--images_folder', type=str, default='imgs', help='images folder')

        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--HR_image_size', type=int, default=128, help='input high resolution image size')
        # self._parser.add_argument('--LR_image_size', type=int, default=32, help='input low resolution image size')

        self._parser.add_argument('--expression_type', type=int, default=7, help='# expression types')
        self._parser.add_argument('--subject_type', type=int, default=118, help='# subject types')

        self._parser.add_argument('--resnet', action='store_true', help='if true, use resnet backbone')
        self._parser.add_argument('--no_C_adv', action='store_true',
                                  help='if true, use the same classifiers for cooperative and adversarial learning')
        self._parser.add_argument('--train_Cross', action='store_true', help='if true, launch cross adversarial training')
        self._parser.add_argument('--train_CrossOnly', action='store_true', help='if true, launch cross adversarial training and no cooperative training')
        self._parser.add_argument('--train_adv', action='store_true', help='if true, choose contrary adversarial training')
        self._parser.add_argument('--train_Rec', action='store_true', help='if true, launch reconstruction training')
        self._parser.add_argument('--no_RecCycle', action='store_true', help='if true, remove cycle-reconstruction training')
        self._parser.add_argument('--train_Gu', action='store_true', help='if true, launch all guidance')
        self._parser.add_argument('--train_Gu_SC', action='store_true', help='if true, launch classified similarity')
        self._parser.add_argument('--train_Gu_LIR', action='store_true', help='if true, launch lir loss')
        self._parser.add_argument('--train_Gu_RSC', action='store_true', help='if true, launch reconstruction similarity')

        self._parser.add_argument('--init', type=str, default='kaiming_normal', help='choose weight initializer, '
                                                                             '[normal | kaiming_normal | kaiming_uniform | xavier_normal | xavier_uniform]')

        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--n_threads_test', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self._parser.add_argument('--use_scheduler', action='store_true', help='enable lr scheduler')
        self._parser.add_argument('--lr_policy', type=str, default='lambda', help='# of epochs at starting learning rate')
        self._parser.add_argument('--lr_change', type=float, default=0.95, help='# of epochs at starting learning rate')
        self._parser.add_argument('--lr_decay_iters', type=int, default=5, help='# of epochs at starting learning rate')
        self._parser.add_argument('--lr_gamma', type=float, default=0.9, help='# of epochs at starting learning rate')

        self._parser.add_argument('--show_time', action='store_true', help='enable verbose time log')

    def parse(self):

        self._opt = self._parser.parse_args()

        # set is train or test
        self._opt.is_train = self.is_train

        if self._opt.train_Gu:
            self._opt.train_Gu_SC = self._opt.train_Gu_LIR = self._opt.train_Gu_RSC = True

        self._opt.expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        self._opt.save_fake_dir = os.path.join(self._opt.expr_dir, self._opt.images_folder)

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        assert torch.cuda.is_available(), "need cuda"
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)
        self._opt.save_results_file = os.path.join(self._opt.expr_dir, self._opt.save_results_file)

        return self._opt

    def _set_and_check_load_epoch(self):
        if os.path.exists(self._opt.expr_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(self._opt.expr_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            elif self._opt.load_epoch != 0:
                found = False
                for file in os.listdir(self._opt.expr_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found at %s' % (self._opt.load_epoch, self._opt.expr_dir)
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(str_id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(int(self._opt.gpu_ids[0]))

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        util.mkdirs(self._opt.expr_dir)
        file_name = os.path.join(self._opt.expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'a') as opt_file:
            now = time.strftime("%c")
            opt_file.write('------------ Options (%s) -------------\n' % now)
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
