import numpy as np
import os
import time
from . import util
from tensorboardX import SummaryWriter


class TBVisualizer:
    def __init__(self, opt):
        self._opt = opt
        self._save_path = os.path.join(opt.checkpoints_dir, opt.name)

        self._log_path = opt.save_results_file
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        self._writer = SummaryWriter(self._tb_path)

        if not os.path.exists(self._save_path):
            util.mkdirs(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self._writer.close()

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors):
        message = '(epoch: %d, it: %d/%d) ' % (epoch, i, iters_per_epoch)
        for k, v in errors.items():
            message += '%s:%.4f ' % (k, v)

        print(message)

        # with open(self._log_path, "a") as log_file:
        #     log_file.write(message+'\n')

    def save_lr(self, lr, epoch):
        with open(self._log_path, "a") as log_file:
            log_file.write('epoch %d current lr: %f\n' % (epoch, lr))