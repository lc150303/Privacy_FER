import time
import torch
import os
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from utils.tb_visualizer import TBVisualizer
from utils import util
import pickle

class Train:
    def __init__(self):

        self._opt = TrainOptions().parse()
        self._data_loader_train = CustomDatasetDataLoader(self._opt, 'train', is_for_train=True)
        self._data_loader_test = CustomDatasetDataLoader(self._opt, 'test', is_for_train=False)
        # self._opt.batch_size //= 2

        self._dataset_train = self._data_loader_train.load_data()
        self._dataset_test = self._data_loader_test.load_data()

        self._dataset_train_size = len(self._data_loader_train)
        self._dataset_test_size = len(self._data_loader_test)
        print('# train images = %d' % self._dataset_train_size)
        print('# test images = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._save_results_file = self._opt.save_results_file
        
        self._tb_visualizer = TBVisualizer(self._opt)

        # if self._opt.load_epoch < 1 and self._opt.pretrain:
        #     self._pretrain()

        self._train()
        # self._save_features()


    def _pretrain(self):
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._total_steps = 0

        self._data_loader_train.set_pretrain()
        for i_epoch in range(self._opt.pretrain_nepochs):
            epoch_start_time = time.time()
            self._pretrain_epoch(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of pretrain epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch+1, self._opt.pretrain_nepochs, time_epoch,
                   time_epoch / 60, time_epoch / 3600))


    def _pretrain_epoch(self, i_epoch):
        for i_pre_batch, data_train_batch in enumerate(self._dataset_train):
            iter_start_time = time.time()
            # pretrain model
            self._model.set_input(data_train_batch)
            self._model.pretrain()

            if i_pre_batch == 0 and (self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De):
                self._generate_imgs(i_epoch, i_pre_batch, data_train_batch['HR_img_path'], data_train_batch['LR_img_path'], True)
            if self._opt.show_time:
                print('iter %d, time:%.3f' % (i_pre_batch, time.time()-iter_start_time))
        print('%d epoch loss' % i_epoch, self._model.get_pretrain_loss())

    def _train(self):
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._total_steps = self._opt.load_epoch * self._dataset_train_size

        self._data_loader_train.set_tune()

        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            epoch_start_time = time.time()

            # train epoch
            self._train_epoch(i_epoch)

            # test epoch
            self._test_epoch(i_epoch)

            # save model
            print('saving the model at the end of epoch %d, iters %d' % (i_epoch, self._total_steps))
            if self._opt.save_model:
                self._model.save(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time

            print('End of epoch %d / %d lr: %f\t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, self._model.get_current_lr(), time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            # update learning rate
            if i_epoch > self._opt.nepochs_no_decay:
                if self._opt.use_scheduler:
                    self._model.update_learning_rate()
                self._save_lr(i_epoch)

    def _train_epoch(self, i_epoch):
        for i_train_batch, data_train_batch in enumerate(self._dataset_train):

            iter_start_time = time.time()

            # train model
            train_batch = {}

            train_batch['HR_img'] = torch.cat((data_train_batch['HR_img_1'],data_train_batch['HR_img_2'].flip(0)),0)
            #print(train_batch['HR_img'].shape)
            train_batch['LR_img'] = torch.cat((data_train_batch['LR_img_1'],data_train_batch['LR_img_2'].flip(0)),0)
            #print(train_batch['LR_img'].shape)
            train_batch['HR_beha'] = torch.cat((data_train_batch['beha_1'],data_train_batch['beha_2'].flip(0)),0)
            #print(train_batch['HR_beha'])
            train_batch['LR_beha'] = train_batch['HR_beha']
            train_batch['HR_subj'] = torch.cat((data_train_batch['subj_1'],data_train_batch['subj_2'].flip(0)),0)
            #print(train_batch['HR_subj'])
            train_batch['LR_subj'] = train_batch['HR_subj']
            #assert 0==1
            data_train_batch['HR_img_path_2'].reverse()
            train_batch['HR_img_path'] = data_train_batch['HR_img_path_1']+data_train_batch['HR_img_path_2']
            data_train_batch['LR_img_path_2'].reverse()
            train_batch['LR_img_path'] = data_train_batch['LR_img_path_1']+data_train_batch['LR_img_path_2']
            self._model.set_input(train_batch)
            self._model.optimize_parameters()

            # update epoch info
            self._total_steps += self._opt.batch_size

            # display terminal
            if i_train_batch % 200 == 0:
                # self._display_terminal(i_epoch, i_train_batch)
                print('epoch', i_epoch, 'iter', i_train_batch)
                if self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De:
                    self._generate_imgs(i_epoch, i_train_batch, train_batch['HR_img_path'], train_batch['LR_img_path'])
            if self._opt.show_time:
                print('iter %d, time:%.3f' % (i_train_batch, time.time()-iter_start_time))
            #break

    def _test_epoch(self, i_epoch):
        self._model.set_eval()
        beha_LR_total = 0
        subj_LR_total = 0
        for i_test_batch, test_batch in enumerate(self._dataset_test):
            LR_beha_logit, LR_subj_logit = self._model.evaluate(test_batch)

            LR_beha_logit = torch.max(LR_beha_logit, 1)[1].data.squeeze().cpu().numpy()
            LR_subj_logit = torch.max(LR_subj_logit, 1)[1].data.squeeze().cpu().numpy()

            beha_label = test_batch['LR_beha'].cpu().numpy()[0]
            subj_label = test_batch['LR_subj'].cpu().numpy()[0]

            if int(LR_beha_logit) == int(beha_label):
                beha_LR_total += 1.0
            if int(LR_subj_logit) == int(subj_label):
                subj_LR_total += 1.0

        self._model.set_train()
        LR_exp_acc, LR_id_acc = beha_LR_total / self._dataset_test_size, subj_LR_total / self._dataset_test_size
        print ("End of epoch %d, the behaviour ACC is %.4f, id ACC is %.4f" % (i_epoch, LR_exp_acc, LR_id_acc))
        with open(self._opt.save_results_file, 'a') as fw:
            fw.write("End of epoch %d, the behaviour ACC %.4f, id ACC is %.4f" % (i_epoch, LR_exp_acc, LR_id_acc))

    def _save_features(self):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        train_dir = os.path.join(expr_dir, 'feature', 'train')
        test_dir = os.path.join(expr_dir, 'feature', 'test')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        i = 0
        for data_train_batch in self._dataset_train:
            for sample in zip(self._model.get_LR_feature(data_train_batch['LR_img_1']).cpu().numpy().tolist(), data_train_batch['beha_1'].numpy().tolist(), data_train_batch['subj_1'].numpy().tolist()):
                pickle.dump(sample, open(os.path.join(train_dir, str(i)+'.pkl'), 'wb'))
                i += 1
            for sample in zip(self._model.get_LR_feature(data_train_batch['LR_img_2']).cpu().numpy().tolist(), data_train_batch['beha_2'].numpy().tolist(), data_train_batch['subj_2'].numpy().tolist()):
                pickle.dump(sample, open(os.path.join(train_dir, str(i) + '.pkl'), 'wb'))
                i += 1

        i=0
        for data_test_batch in self._dataset_test:
            for sample in zip(self._model.get_LR_feature(data_test_batch['LR_img']).cpu().numpy().tolist(), data_test_batch['LR_beha'].numpy().tolist(), data_test_batch['LR_subj'].numpy().tolist()):
                pickle.dump(sample, open(os.path.join(test_dir, str(i)  + '.pkl'), 'wb'))
                i += 1

        # pickle.dump(train_features, open(os.path.join(expr_dir, 'train_features.pkl'), 'wb'))
        # pickle.dump(test_features, open(os.path.join(expr_dir, 'test_features.pkl'), 'wb'))
        print('features saved')


    def _display_terminal(self, i_epoch, i_train_batch):
        errors = self._model.get_current_errors()
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors)

    def _generate_imgs(self, i_epoch, i_train_batch, HR_img_path, LR_img_path, pretrain=False):
        # get current fake image
        HR_fake_img, LR_fake_img = self._model.get_current_fake_img()
        print('save HR', HR_fake_img.shape, 'HR', LR_fake_img.shape)
        for index in range(HR_fake_img.shape[0] if HR_fake_img.shape[0] < 4 else 4):
            # select one image - see "util.tensor2im"
            vis_HR_fake_img = util.tensor2im(HR_fake_img.data, idx=index)
            vis_LR_fake_img = util.tensor2im(LR_fake_img.data, idx=index)
            vis_HR_img_path = HR_img_path[index]
            vis_LR_img_path = LR_img_path[index]

            save_HR_path = "HR_%d_%d_%s" % (i_epoch, i_train_batch, os.path.basename(vis_HR_img_path))
            save_LR_path = "LR_%d_%d_%s" % (i_epoch, i_train_batch, os.path.basename(vis_LR_img_path))
            if pretrain:
                save_HR_path = os.path.join(self._opt.save_fake_dir, 'pretrain', save_HR_path)
                save_LR_path = os.path.join(self._opt.save_fake_dir, 'pretrain', save_LR_path)
            else:
                save_HR_path = os.path.join(self._opt.save_fake_dir, save_HR_path)
                save_LR_path = os.path.join(self._opt.save_fake_dir, save_LR_path)
            
            util.save_image(vis_HR_fake_img, save_HR_path)
            util.save_image(vis_LR_fake_img, save_LR_path)
            #print (vis_HR_img_path)
            #print (vis_LR_img_path)
            #print (vis_HR_fake_img.shape)
            #print (vis_LR_fake_img.shape)

    def _save_lr(self, i_epoch):
        self._tb_visualizer.save_lr(self._model.get_current_lr(), i_epoch)


        
if __name__ == "__main__":
    Train()
