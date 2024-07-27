import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
from .models import BaseModel
from networks.networks import NetworksFactory
import utils.util as util
import os
import numpy as np

"""
Low Resolution Privacy Protect Networks
"""
class LRPPNet(BaseModel):
    def __init__(self, opt):
        super(LRPPNet, self).__init__(opt)
        self._name = 'LRPPNet'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars(opt)

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses(opt)

        # ------- for generative loss -------
        if self._opt.train_Dis:
            self._real_labels =self._Tensor(np.ones(self._opt.batch_size)).detach()
            self._fake_labels =self._Tensor(np.zeros(self._opt.batch_size)).detach()
            #print (self._real_labels.shape)

    def _init_create_networks(self):
        # HR&LR beharession encoder - weight sharing
        self._En_h = self._create_branch('En_h')
        self._En_h.init_weights(self._opt.init)
        self._opt.feature_size = self._En_h.output_size
        self._opt.feature_dim = self._En_h.output_dim

        # HR&LR id encoder - weight sharing
        self._En_l = self._create_branch('En_l')
        self._En_l.init_weights(self._opt.init)

        # behaviour classifier
        self._C_e = self._create_branch('C_e')
        self._C_e.init_weights(self._opt.init)

        if self._opt.train_Cl:
            self._C_l = self._create_branch('C_l')
            self._C_l.init_weights(self._opt.init)
        else:
            self._C_l = self._C_e

        #self._attacker3 = self._create_branch('a3')
        #self._attacker3.init_weights()

        # id classifier
        self._C_id = self._create_branch('C_id')
        self._C_id.init_weights(self._opt.init)

        # ------- for adversarial loss -------
        if self._opt.train_Cyc or self._opt.train_Dis or self._opt.train_De:
            # decoder
            self._De = self._create_branch('De')
            self._De.init_weights(self._opt.init)

            # Discriminator for id and real/fake
            if self._opt.train_Dis:
                self._Dis_id_r = self._create_branch('Dis_id_r')
                self._Dis_id_r.init_weights(self._opt.init)

                # Discriminator for behaviours
                self._Dis_e = self._create_branch('Dis_e')
                self._Dis_e.init_weights(self._opt.init)

        if torch.cuda.is_available():
            self._En_h.cuda()
            self._En_l.cuda()
            self._C_e.cuda()
            self._C_id.cuda()
            self._C_l.cuda()
            # ------- for adversarial loss -------
            if self._opt.train_Cyc or self._opt.train_Dis or self._opt.train_De:
                self._De.cuda()
                if self._opt.train_Dis:
                    self._Dis_e.cuda()
                    self._Dis_id_r.cuda()

    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name, self._opt)

    def _init_train_vars(self, opt):
        self._current_lr_En = self._opt.lr_En
        self._current_lr_C = self._opt.lr_C
        self._current_lr_De = self._opt.lr_De
        self._current_lr_Dis = self._opt.lr_Dis

        # initialize optimizers
        self._optimizer_En_h = torch.optim.Adam([{'params': self._En_h.parameters(), 'initial_lr':self._current_lr_En}],
                                                lr=self._current_lr_En, betas=(self._opt.En_adam_b1, self._opt.En_adam_b2))
        self._optimizer_En_l = torch.optim.Adam([{'params': self._En_l.parameters(), 'initial_lr':self._current_lr_En}],
                                                lr=self._current_lr_En, betas=(self._opt.En_adam_b1, self._opt.En_adam_b2))

        self._optimizer_C = torch.optim.Adam([{'params': self._C_id.parameters(), 'initial_lr':self._current_lr_C},
                                                 {'params': self._C_e.parameters(), 'initial_lr':self._current_lr_C}
                                                 ], lr=self._current_lr_C,
                                                 betas=(self._opt.C_adam_b1, self._opt.C_adam_b2))

        if self._opt.train_Cl:
            self._optimizer_C_l = torch.optim.Adam([{'params': self._C_l.parameters(), 'initial_lr':self._current_lr_C}],
                                                lr=self._current_lr_C, betas=(self._opt.C_adam_b1, self._opt.C_adam_b2))

        # ------- for adversarial loss -------
        if self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De:
            self._optimizer_De = torch.optim.Adam([{'params': self._De.parameters(), 'initial_lr':self._current_lr_De}],
                                                  lr=self._current_lr_De, betas=(self._opt.De_adam_b1, self._opt.De_adam_b2))
            if  self._opt.train_Dis:
                self._optimizer_Dis = torch.optim.Adam([{'params': self._Dis_e.parameters(), 'initial_lr':self._current_lr_Dis},
                                                         {'params': self._Dis_id_r.parameters(), 'initial_lr':self._current_lr_Dis}
                                                        ], lr=self._current_lr_Dis,
                                                       betas=(self._opt.Dis_adam_b1, self._opt.Dis_adam_b2))

        if self._opt.use_scheduler:
            self._lr_scheduler_En_h = self._get_scheduler(self._optimizer_En_h, opt)
            self._lr_scheduler_En_l = self._get_scheduler(self._optimizer_En_l, opt)
            self._lr_scheduler_C = self._get_scheduler(self._optimizer_C, opt)
            if self._opt.train_Cl:
                self._lr_scheduler_C_l = self._get_scheduler(self._optimizer_C_l, opt)
            if self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De:
                self._lr_scheduler_De = self._get_scheduler(self._optimizer_De, opt)
                if  self._opt.train_Dis:
                    self._lr_scheduler_Dis = self._get_scheduler(self._optimizer_Dis, opt)

    def _init_prefetch_inputs(self):
        self._input_HR_img = self._Tensor(self._opt.batch_size, 3, self._opt.HR_image_size, self._opt.HR_image_size)
        self._input_LR_img = self._Tensor(self._opt.batch_size, 3, self._opt.LR_image_size, self._opt.LR_image_size)
        self._input_HR_beha = self._LongTensor(self._opt.batch_size, 1)
        self._input_LR_beha = self._LongTensor(self._opt.batch_size, 1)
        self._input_HR_id = self._LongTensor(self._opt.batch_size, 1)
        self._input_LR_id = self._LongTensor(self._opt.batch_size, 1)

    def _init_losses(self, opt):
        # define loss function
        self._cross_entropy = torch.nn.CrossEntropyLoss().cuda()
        self._mse_loss = torch.nn.MSELoss().cuda()
        self._lir_loss = LossInequalityRegulation(opt).cuda()

    def set_input(self, input):
        self._input_HR_img.resize_(input['HR_img'].size()).copy_(input['HR_img'])
        self._input_LR_img.resize_(input['LR_img'].size()).copy_(input['LR_img'])
        self._input_HR_beha.resize_(input['HR_beha'].size()).copy_(input['HR_beha'])
        self._input_LR_beha.resize_(input['LR_beha'].size()).copy_(input['LR_beha'])
        self._input_HR_id.resize_(input['HR_subj'].size()).copy_(input['HR_subj'])
        self._input_LR_id.resize_(input['LR_subj'].size()).copy_(input['LR_subj'])

        if torch.cuda.is_available():
            self._input_HR_img = self._input_HR_img.cuda('cuda:'+self._gpu_ids[0], non_blocking=True)
            self._input_LR_img = self._input_LR_img.cuda('cuda:'+self._gpu_ids[0], non_blocking=True)
            self._input_HR_beha = self._input_HR_beha.cuda('cuda:'+self._gpu_ids[0], non_blocking=True)
            self._input_LR_beha = self._input_LR_beha.cuda('cuda:'+self._gpu_ids[0], non_blocking=True)
            self._input_HR_id = self._input_HR_id.cuda('cuda:'+self._gpu_ids[0], non_blocking=True)
            self._input_LR_id = self._input_LR_id.cuda('cuda:'+self._gpu_ids[0], non_blocking=True)
            #print (self._input_HR_img.shape)

    def set_train(self):
        self._En_l.train()
        self._En_h.train()
        self._C_id.train()
        self._C_e.train()
        # ------- for generative adversarial training -------
        if self._opt.train_Cyc or self._opt.train_Dis or self._opt.train_De:
            self._De.train()
            if self._opt.train_Dis:
                self._Dis_e.train()
                self._Dis_id_r.train()
        self._is_train = True

    def set_eval(self):
        self._En_l.eval()
        self._En_h.eval()
        self._C_id.eval()
        self._C_e.eval()
        # ------- for generative adversarial training -------
        if self._opt.train_Cyc or self._opt.train_Dis or self._opt.train_De:
            self._De.eval()
            if self._opt.train_Dis:
                self._Dis_e.eval()
                self._Dis_id_r.eval()
        self._is_train = False

    def pretrain(self):
        self._HR_img = self._input_HR_img
        self._LR_img = self._input_LR_img
        self._HR_beha = self._input_HR_beha
        self._LR_beha = self._input_LR_beha
        self._HR_id = self._input_HR_id
        self._LR_id = self._input_LR_id

        self._forward_En()

        HR_loss, LR_loss = self._forward_pretrain_C()
        self._optimizer_En_h.zero_grad()
        self._optimizer_C.zero_grad()
        HR_loss.backward(retain_graph=True)
        self._optimizer_En_h.step()
        self._optimizer_C.step()


        if self._opt.train_Cyc or self._opt.train_De or self._opt.train_Dis:
            self._optimizer_En_l.zero_grad()
            LR_loss.backward(retain_graph=True)
            self._optimizer_En_l.step()

            loss_De = self._forward_pretrain_De()
            self._optimizer_De.zero_grad()
            self._optimizer_En_h.zero_grad()
            self._optimizer_En_l.zero_grad()
            loss_De.backward()
            self._optimizer_De.step()
            self._optimizer_En_h.step()
            self._optimizer_En_l.step()
        else:
            self._optimizer_En_l.zero_grad()
            LR_loss.backward()
            self._optimizer_En_l.step()

    def _forward_pretrain_C(self):

        # C_e(f_h) & C_e(f_l)
        HR_beha_logit = self._C_e.forward(self._HR_feature_e)
        LR_beha_logit = self._C_l.forward(self._LR_feature_e)

        HR_id_logit = self._C_id.forward(self._HR_feature_id)

        HR_loss = self._cross_entropy(HR_beha_logit, self._HR_beha) + \
                            self._cross_entropy(HR_id_logit, self._HR_id)

        LR_loss = self._cross_entropy(LR_beha_logit, self._LR_beha) + \
                            self._mse_loss(LR_beha_logit, HR_beha_logit)*self._opt.L_cls_sim
        self._loss_pretrain_C = HR_loss + LR_loss

        return HR_loss, LR_loss


    def _forward_pretrain_De(self):
        self._HR_reconstruction = self._De.forward(self._HR_featuremap_id, self._HR_featuremap_e)
        self._LR_reconstruction = self._De.forward(self._HR_featuremap_id, self._LR_featuremap_e)

        self._loss_pretrain_De = self._mse_loss(self._HR_reconstruction, self._HR_img) + \
                              self._mse_loss(self._LR_reconstruction, self._LR_img)
        return self._loss_pretrain_De

    def get_loss(self):
        if self._opt.train_Cyc or self._opt.train_De or self._opt.train_Dis:
            return self._loss_pretrain_De.item(), self._loss_pretrain_C.item()
        else:
            return self._loss_pretrain_C.item()

    def optimize_parameters(self):
        if self._is_train:
            # convert tensor to variables
            self._HR_img = self._input_HR_img
            self._LR_img = self._input_LR_img
            self._HR_beha = self._input_HR_beha
            self._LR_beha = self._input_LR_beha
            self._HR_id = self._input_HR_id
            self._LR_id = self._input_LR_id

            # extract features
            # print('f En')
            self._forward_En()

            # train C_id C_e
            # print('f C')
            loss_C_HR, loss_C_LR, loss_cross = self._forward_C()
            self._optimizer_En_l.zero_grad()
            self._optimizer_C.zero_grad()
            loss_C_LR.backward()
            self._optimizer_En_l.step()
            self._optimizer_C.step()

    def _forward_En(self):
        # extract feature from HR images and LR images
        self._HR_feature_id, self._HR_feature_e, self._HR_featuremap_id, self._HR_featuremap_e = self._En_h.forward(self._HR_img)
        self._LR_feature_e, self._LR_featuremap_e = self._En_l.forward(self._LR_img)

    def _forward_C(self):

        # C_e(f_h) & C_e(f_l)
        LR_beha_logit = self._C_l.forward(self._LR_feature_e)

        # C_id(f_h)
        LR_id_logit = self._C_id.forward(self._LR_feature_e)

        # cross-entropy loss for C_e(f_h)
        self._LR_loss_C = self._cross_entropy(LR_beha_logit, self._HR_beha) + \
                            self._cross_entropy(LR_id_logit, self._HR_id)
        # ------- for cross advers arial loss -------
        self._cross_loss = 0
        self._HR_loss_C = 0

        # print('C', self._HR_loss_C, self._LR_loss_C)
        return self._HR_loss_C, self._LR_loss_C, self._cross_loss

    def update_learning_rate(self):
        self._lr_scheduler_En_h.step()
        self._lr_scheduler_En_l.step()
        self._lr_scheduler_C.step()

        if self._opt.train_Cl:
            self._lr_scheduler_C_l.step()

        if self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De:
            self._lr_scheduler_De.step()
            if self._opt.train_Dis:
                self._lr_scheduler_Dis.step()

    def evaluate(self, input):
        self._input_HR_img.resize_(input['HR_img'].size()).copy_(input['HR_img'])
        self._input_LR_img.resize_(input['LR_img'].size()).copy_(input['LR_img'])
        HR_img = self._input_HR_img
        LR_img = self._input_LR_img
        with torch.no_grad():
            LR_feature_e, _ = self._En_l.forward(LR_img)
            LR_beha_logit = self._C_e.forward(LR_feature_e)
            LR_subj_logit = self._C_id.forward(LR_feature_e)
        return LR_beha_logit, LR_subj_logit

    def get_current_lr(self):
        return self._optimizer_En_h.param_groups[0]['lr']

    def get_LR_feature(self, img):
        img = self._input_LR_img.resize_(img.size()).copy_(img)
        with torch.no_grad():
            LR_feature_e, _ = self._En_l.forward(img)
        return LR_feature_e

    def get_current_errors(self):
        loss_dict = OrderedDict([('HR_loss_C', self._HR_loss_C.item()),
                                 ('LR_loss_C', self._LR_loss_C.item())
                                 ])
        # ------- for generative adversarial loss -------
        if self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De:
            loss_dict['loss_De'] = self._loss_De.item()
            if self._opt.train_Dis:
                loss_dict['loss_Dis_e'] = self._loss_Dis_e.item()
                loss_dict['loss_Dis_id_r'] = self._loss_Dis_id_r.item()
            if self._opt.train_Cyc:
                loss_dict['loss_Cyc'] = self._loss_Cyc.item()

        return loss_dict

    def print_current_label(self):
        if self._opt.train_Dis:
            print('t/f label:', self._real_labels, self._fake_labels)

    def save(self, label):
        # save networks
        self._save_network(self._En_l, 'En_l', label)
        self._save_network(self._En_h, 'En_h', label)
        self._save_network(self._C_e, 'C_e', label)
        self._save_network(self._C_id, 'C_id', label)

        self._save_optimizer(self._optimizer_En_h, 'En_h', label)
        self._save_optimizer(self._optimizer_En_l, 'En_l', label)
        self._save_optimizer(self._optimizer_C, 'C', label)

        if self._opt.train_Cl:
            self._save_network(self._C_l, 'C_l', label)
            self._save_optimizer(self._optimizer_C_l, 'C_l', label)

        # ------- for GAN -------
        if self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De:
            self._save_network(self._De, 'De', label)
            self._save_optimizer(self._optimizer_De, 'De', label)
            if self._opt.train_Dis:
                self._save_network(self._Dis_id_r, 'Dis_id_r', label)
                self._save_network(self._Dis_e, 'Dis_e', label)
                self._save_optimizer(self._optimizer_Dis, 'Dis', label)

    def load(self):
        load_epoch = self._opt.load_epoch
        self._load_network(self._En_l, 'En_l', load_epoch)
        self._load_network(self._En_h, 'En_h', load_epoch)
        self._load_network(self._C_e, 'C_e', load_epoch)
        self._load_network(self._C_id, 'C_id', load_epoch)

        self._load_optimizer(self._optimizer_En_h, 'En_h', load_epoch)
        self._load_optimizer(self._optimizer_En_l, 'En_l', load_epoch)
        self._load_optimizer(self._optimizer_C, 'C', load_epoch)

        if self._opt.train_Cl:
            self._load_network(self._C_l, 'C_l', load_epoch)
            self._load_optimizer(self._optimizer_C_l, 'C_l', load_epoch)

        # ------- for GAN -------
        if self._opt.train_Dis or self._opt.train_Cyc or self._opt.train_De:
            self._load_network(self._De, 'De', load_epoch)
            self._load_optimizer(self._optimizer_De, 'De', load_epoch)
            if self._opt.train_Dis:
                self._load_network(self._Dis_id_r, 'Dis_id_r', load_epoch)
                self._load_network(self._Dis_e, 'Dis_e', load_epoch)
                self._load_optimizer(self._optimizer_Dis, 'Dis', load_epoch)

    # ------- for generator -------
    def get_current_fake_img(self):
        return self._HR_reconstruction, self._LR_reconstruction


class LossInequalityRegulation(nn.Module):
    def __init__(self, opt):
        super(LossInequalityRegulation, self).__init__()
        self._zeos = torch.zeros((opt.batch_size, opt.expression_type)).cuda()

    def forward(self, loss_l, loss_h):
        inequality = loss_l - loss_h
        #print(inequality)
        l_lir = torch.max(torch.stack((inequality, self._zeos), dim=0), dim=0)[0]
        #print(llir.sum())
        return l_lir.sum()
