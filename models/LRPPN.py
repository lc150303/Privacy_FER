import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
# from torchstat import stat

from networks.networks import NetworksFactory
from .models import BaseModel

"""
Low Resolution Privacy Protect Networks
"""


class LRPPNet(BaseModel):
    def __init__(self, opt):
        super(LRPPNet, self).__init__(opt)
        self._name = 'LRPPNet'
        self._train_Gu = opt.train_Gu_SC or opt.train_Gu_LIR or opt.train_Gu_RSC

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_optimizer_and_scheduler(opt)

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()
        else:
            print('use new model', self._is_train, self._opt.load_epoch)

        # init
        self._init_losses(opt)

        self.mean_exp = self._Tensor(np.ones((self._opt.batch_size, self._opt.expression_type))/self._opt.expression_type)
        self.mean_id = self._Tensor(np.ones((self._opt.batch_size, self._opt.subject_type))/self._opt.subject_type)

        self.feature_exp_target = []
        for i in range(opt.expression_type):
            self.feature_exp_target.append(self._Tensor(
                np.zeros((1, 2))
            ))

    def _init_create_networks(self):
        self.networks = {}
        opt = self._opt

        # HR encoder
        En_h = self._create_branch('En_h')
        # stat(En_h, input_size=(3, 128, 128))
        En_h.init_weights(opt.init)
        print('En out_size', En_h.output_size, ', En out_dim', En_h.output_dim)
        opt.feature_size = En_h.output_size
        opt.feature_dim = En_h.output_dim
        self.networks['En_h'] = En_h

        # LR encoder
        En_l = self._create_branch('En_l')
        # stat(En_h, input_size=(3, 128, 128))
        En_l.init_weights(opt.init)
        self.networks['En_l'] = En_l

        """
        建立分类器，其中若 train_CrossOnly 则只训练 cross-adversarial loss
        否则必然有正常分类 loss，此时
           若 train_Cross 则同时有正常分类 loss 和  cross-adversarial loss
              此时两个 loss 有共享分类器和独立分类器 2 种情况
        """
        if opt.train_CrossOnly:
            # behaviour classifier
            C_e_adv = self._create_branch('C_e_adv')
            C_e_adv.init_weights(opt.init)
            self.networks['C_e_adv'] = C_e_adv

            # id classifier
            C_id_adv = self._create_branch('C_id_adv')
            C_id_adv.init_weights(opt.init)
            self.networks['C_id_adv'] = C_id_adv
        else:
            # behaviour classifier
            C_e = self._create_branch('C_e')
            # summary(C_e, input_size=(En_h.output_size//2, ), device='cpu')
            C_e.init_weights(opt.init)
            self.networks['C_e'] = C_e

            # id classifier
            C_id = self._create_branch('C_id')
            C_id.init_weights(opt.init)
            self.networks['C_id'] = C_id

            if opt.no_C_adv:
                self.networks['C_e_adv'] = C_e
                self.networks['C_id_adv'] = C_id
            else:
                # behaviour classifier
                C_e_adv = self._create_branch('C_e_adv')
                C_e_adv.init_weights(opt.init)
                self.networks['C_e_adv'] = C_e_adv

                # id classifier
                C_id_adv = self._create_branch('C_id_adv')
                C_id_adv.init_weights(opt.init)
                self.networks['C_id_adv'] = C_id_adv

        # ------- for Reconstruction -------
        if opt.train_Rec:
            # decoder
            De = self._create_branch('De')
            De.init_weights(opt.init)
            self.networks['De'] = De

        if torch.cuda.is_available():
            for name in self.networks:
                self.networks[name] = self.networks[name].cuda()

    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name, self._opt)

    def _init_optimizer_and_scheduler(self, opt):
        self.current_lrs = {}
        self.current_lrs['En'] = opt.lr_En
        self.current_lrs['C'] = opt.lr_C
        self.current_lrs['De'] = opt.lr_De

        self.optimizers = {}
        # initialize optimizers
        self.optimizers['En_h'] = torch.optim.Adam([{'params': self.networks['En_h'].parameters(),
                                                     'initial_lr': self.current_lrs['En']}],
                                                   lr=self.current_lrs['En'], betas=(opt.En_adam_b1, opt.En_adam_b2))
        self.optimizers['En_l'] = torch.optim.Adam([{'params': self.networks['En_l'].parameters(),
                                                     'initial_lr': self.current_lrs['En']}],
                                                   lr=self.current_lrs['En'], betas=(opt.En_adam_b1, opt.En_adam_b2))

        if opt.train_CrossOnly:
            self.optimizers['C_id_adv'] = torch.optim.Adam([{'params': self.networks['C_id_adv'].parameters(),
                                                             'initial_lr': self.current_lrs['C']}],
                                                           lr=self.current_lrs['C'], betas=(opt.C_adam_b1, opt.C_adam_b2))
            self.optimizers['C_e_adv'] = torch.optim.Adam([{'params': self.networks['C_e_adv'].parameters(),
                                                            'initial_lr': self.current_lrs['C']}],
                                                          lr=self.current_lrs['C'],
                                                          betas=(opt.C_adam_b1, opt.C_adam_b2))
        else:
            self.optimizers['C_e'] = torch.optim.Adam([{'params': self.networks['C_e'].parameters(),
                                                        'initial_lr': self.current_lrs['C']}],
                                                      lr=self.current_lrs['C'], betas=(opt.C_adam_b1, opt.C_adam_b2))
            self.optimizers['C_id'] = torch.optim.Adam([{'params': self.networks['C_id'].parameters(),
                                                         'initial_lr': self.current_lrs['C']}],
                                                       lr=self.current_lrs['C'], betas=(opt.C_adam_b1, opt.C_adam_b2))

            if opt.no_C_adv:
                self.optimizers['C_id_adv'] = self.optimizers['C_id']
                self.optimizers['C_e_adv'] = self.optimizers['C_e']
            else:
                self.optimizers['C_id_adv'] = torch.optim.Adam([{'params': self.networks['C_id_adv'].parameters(),
                                                                 'initial_lr': self.current_lrs['C']}],
                                                               lr=self.current_lrs['C'], betas=(opt.C_adam_b1, opt.C_adam_b2))
                self.optimizers['C_e_adv'] = torch.optim.Adam([{'params': self.networks['C_e_adv'].parameters(),
                                                                'initial_lr': self.current_lrs['C']}],
                                                              lr=self.current_lrs['C'],
                                                              betas=(opt.C_adam_b1, opt.C_adam_b2))

        # ------- for reconstruction loss -------
        if opt.train_Rec:
            self.optimizers['De'] = torch.optim.Adam([{'params': self.networks['De'].parameters(),
                                                       'initial_lr': self.current_lrs['De']}],
                                                     lr=self.current_lrs['De'], betas=(opt.De_adam_b1, opt.De_adam_b2))

        if opt.use_scheduler:
            self.lr_schedulers = {}
            for name in self.optimizers:
                self.lr_schedulers[name] = self._get_scheduler(self.optimizers[name], opt)

            if not opt.train_CrossOnly and opt.train_Cross and opt.no_C_adv:
                try:
                    del self.lr_schedulers['C_id_adv']
                    del self.lr_schedulers['C_e_adv']
                except:
                    pass

    def _init_losses(self, opt):
        # define loss function
        self._cross_entropy = torch.nn.CrossEntropyLoss().cuda()
        self._mse_loss = torch.nn.MSELoss().cuda()
        self._lir_loss = LossInequalityRegulation(opt).cuda()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()

        self.losses = {}

    def set_input(self, input):
        self._HR_img = input['HR_img'].cuda(non_blocking=True)
        self._LR_img = input['LR_img'].cuda(non_blocking=True)
        self._HR_beha = input['HR_beha'].cuda(non_blocking=True)
        self._LR_beha = input['LR_beha'].cuda(non_blocking=True)
        self._HR_id = input['HR_subj'].cuda(non_blocking=True)
        self._LR_id = input['LR_subj'].cuda(non_blocking=True)

    def set_train(self):
        for name in self.networks:
            self.networks[name].train()
        self._is_train = True

    def set_eval(self):
        for name in self.networks:
            self.networks[name].eval()
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
        self.optimizers['En_h'].zero_grad()
        self.optimizers['C_id'].zero_grad()
        HR_loss.backward(create_graph=True)
        self.optimizers['En_h'].step()
        self.optimizers['C_id'].step()

        if self._opt.train_Cyc or self._opt.train_De or self._opt.train_Dis:
            self._optimizer_En_l.zero_grad()
            LR_loss.backward(create_graph=True)
            self._optimizer_En_l.step()

            loss_De = self._forward_pretrain_De()
            self._optimizer_De.zero_grad()
            self.optimizers['En_h'].zero_grad()
            self._optimizer_En_l.zero_grad()
            loss_De.backward()
            self._optimizer_De.step()
            self.optimizers['En_h'].step()
            self._optimizer_En_l.step()
        else:
            self._optimizer_En_l.zero_grad()
            LR_loss.backward()
            self._optimizer_En_l.step()

    def _forward_pretrain_C(self):

        # C_e(f_h) & C_e(f_l)
        HR_beha_logit = self.networks['C_e'].forward(self._HR_feature_e)
        LR_beha_logit = self.networks['C_e'].forward(self._LR_feature_e)

        HR_id_logit = self.networks['C_id'].forward(self._HR_feature_id)

        HR_loss = self._cross_entropy(HR_beha_logit, self._HR_beha) + \
                  self._cross_entropy(HR_id_logit, self._HR_id)

        LR_loss = self._cross_entropy(LR_beha_logit, self._LR_beha)
        self._loss_pretrain_C = HR_loss + LR_loss

        return HR_loss, LR_loss

    def _forward_pretrain_De(self):
        self._HR_reconstruction = self.networks['De'].forward(self._HR_featuremap_id, self._HR_featuremap_e)
        self._LR_reconstruction = self.networks['De'].forward(self._HR_featuremap_id, self._LR_featuremap_e)

        self._loss_pretrain_De = self._mse_loss(self._HR_reconstruction, self._HR_img) + \
                                 self._mse_loss(self._LR_reconstruction, self._LR_img)
        return self._loss_pretrain_De

    def get_pretrain_loss(self):
        if self._opt.train_Cyc or self._opt.train_De or self._opt.train_Dis:
            return self._loss_pretrain_De.item(), self._loss_pretrain_C.item()
        else:
            return self._loss_pretrain_C.item()

    def optimize_parameters(self):
        if self._is_train:
            opt = self._opt

            # cooperative
            if not opt.train_CrossOnly:
                # extract features
                self._forward_En()
                loss_C_HR, loss_C_LR = self._forward_C_coop()
                self.optimizers['En_h'].zero_grad()
                self.optimizers['En_l'].zero_grad()
                self.optimizers['C_id'].zero_grad()
                self.optimizers['C_e'].zero_grad()
                loss_C_HR.backward()
                loss_C_LR.backward()
                self.optimizers['En_h'].step()
                self.optimizers['En_l'].step()
                self.optimizers['C_id'].step()
                self.optimizers['C_e'].step()

            # adversarial
            with torch.no_grad():
                self._forward_En()
            loss_cross = self._forward_C_cross()  # for update classifiers
            self.optimizers['C_id_adv'].zero_grad()
            self.optimizers['C_e_adv'].zero_grad()
            loss_cross.backward()
            self.optimizers['C_id_adv'].step()
            self.optimizers['C_e_adv'].step()
            if opt.train_CrossOnly or opt.train_Cross:
                # extract features
                self._forward_En()
                loss_adv = self._forward_C_adv()        # for update encoder
                self.optimizers['En_h'].zero_grad()
                self.optimizers['En_l'].zero_grad()
                loss_adv.backward()
                self.optimizers['En_h'].step()
                self.optimizers['En_l'].step()

            # reconstruction
            if opt.train_Rec:
                # extract features
                self._forward_En()
                loss_recon = self._forward_De()
                self.optimizers['De'].zero_grad()
                self.optimizers['En_h'].zero_grad()
                loss_recon.backward()
                self.optimizers['En_h'].step()
                self.optimizers['De'].step()

            # guidance
            if self._train_Gu:
                # extract features
                self._forward_En()
                loss_guidance = self._forward_guidance()

                self.optimizers['En_l'].zero_grad()
                loss_guidance.backward()
                self.optimizers['En_l'].step()

    def _forward_En(self):
        # extract feature from HR images and LR images
        self._HR_feature_id, self._HR_feature_e, self._HR_featuremap_id, self._HR_featuremap_e = self.networks['En_h'].forward(
            self._HR_img)
        self._LR_feature_e, self._LR_featuremap_e = self.networks['En_l'].forward(self._LR_img)

    def _forward_C_coop(self):
        """ cooperatively update classifiers and encoders """
        # C_e(f_h) & C_e(f_l)
        HR_beha_logit = self.networks['C_e'].forward(self._HR_feature_e)
        LR_beha_logit = self.networks['C_e'].forward(self._LR_feature_e)

        # C_id(f_h)
        HR_id_logit = self.networks['C_id'].forward(self._HR_feature_id)

        # cross-entropy loss for C_e(f_h)
        self.losses['HR_C'] = self._cross_entropy(HR_beha_logit, self._HR_beha) + \
                              self._cross_entropy(HR_id_logit, self._HR_id)

        self.losses['LR_C'] = self._cross_entropy(LR_beha_logit, self._LR_beha)

        # print('for C',torch.max(self._HR_loss_C, self._cross_loss_e_id + self._cross_loss_id_e))
        # print('C', self._HR_loss_C, self._LR_loss_C)
        return self.losses['HR_C'], self.losses['LR_C']

    def _forward_C_cross(self):
        """ crossly feed features to classifiers, update classifiers only """
        cross_logit_id_e = self.networks['C_e_adv'].forward(self._HR_feature_id.detach())
        cross_loss_id_e_HR = self._cross_entropy(cross_logit_id_e, self._HR_beha)

        cross_logit_e_id = self.networks['C_id_adv'].forward(self._HR_feature_e.detach())
        cross_loss_e_id_HR = self._cross_entropy(cross_logit_e_id, self._HR_id)

        cross_logit_e_id = self.networks['C_id_adv'].forward(self._LR_feature_e.detach())
        cross_loss_e_id_LR = self._cross_entropy(cross_logit_e_id, self._HR_id)

        self.losses['cross_C'] = (cross_loss_e_id_HR + cross_loss_id_e_HR + cross_loss_e_id_LR) * self._opt.L_cross

        return self.losses['cross_C']

    def _forward_C_adv(self):
        """ crossly feed features to classifiers, update encoder only """
        cross_logit_HR_e_id = self.networks['C_id_adv'].forward(self._HR_feature_e)
        cross_logit_LR_e_id = self.networks['C_id_adv'].forward(self._LR_feature_e)
        cross_logit_id_e = self.networks['C_e_adv'].forward(self._HR_feature_id)

        if self._opt.train_adv:
            adv_loss_e_id_HR = -self._cross_entropy(cross_logit_HR_e_id, self._HR_id)
            adv_loss_e_id_LR = -self._cross_entropy(cross_logit_LR_e_id, self._HR_id)
            adv_loss_id_e = -self._cross_entropy(cross_logit_id_e, self._HR_beha)
        else:
            adv_loss_e_id_HR = self._mse_loss(self.logsoftmax(cross_logit_HR_e_id), self.mean_id)
            adv_loss_e_id_LR = self._mse_loss(self.logsoftmax(cross_logit_LR_e_id), self.mean_id)
            adv_loss_id_e = self._mse_loss(self.logsoftmax(cross_logit_id_e), self.mean_exp)

        self.losses['En_hl_adv'] = self._opt.L_adv * (adv_loss_e_id_HR + adv_loss_e_id_LR + adv_loss_id_e)

        return self.losses['En_hl_adv']

    def _forward_De(self):
        """ forward HR reconstruction and cycle-reconstruction """
        # use different id map
        flip_HR_map_id = self._HR_featuremap_id.flip(0)
        self._flip_HR_id = self._HR_id.flip(0)

        self._HR_reconstruction = self.networks['De'].forward(flip_HR_map_id, self._HR_featuremap_e)
        self.losses['HR_rec'] = self._mse_loss(self._HR_reconstruction, self._HR_img.flip(0))

        """ don't do cycle-reconstruction """
        if self._opt.no_RecCycle:
            return self.losses['HR_rec']

        # cyc_feature_id, cyc_feature_e, cyc_featuremap_id, cyc_featuremap_e = self.networks['En_h'].forward(self._HR_reconstruction)
        _, _, _, cyc_featuremap_e = self.networks['En_h'].forward(self._HR_reconstruction)
        self._cyc_imgs = self.networks['De'].forward(self._HR_featuremap_id, cyc_featuremap_e)
        self.losses['HR_rec'] += self._mse_loss(self._cyc_imgs, self._HR_img) * self._opt.L_cyc

        return self.losses['HR_rec']

    def _forward_guidance(self):
        opt = self._opt
        loss_Gu = 0     # guidance

        if opt.train_Gu_SC or opt.train_Gu_LIR:
            HR_beha_logit = self.networks['C_e'].forward(self._HR_feature_e).detach()
            LR_beha_logit = self.networks['C_e'].forward(self._LR_feature_e)

            if opt.train_Gu_SC:
                self.losses['Gu_sc'] = self._mse_loss(LR_beha_logit, HR_beha_logit) * self._opt.L_cls_sim
                loss_Gu += self.losses['Gu_sc']
            if opt.train_Gu_LIR:
                loss_HR = self._cross_entropy(HR_beha_logit, self._HR_beha)
                loss_LR = self._cross_entropy(LR_beha_logit, self._LR_beha)
                # loss_Gu += self._lir_loss(loss_LR, loss_HR) * self._opt.L_lir

        if opt.train_Gu_RSC:
            flip_HR_map_id = self._HR_featuremap_id.flip(0)
            self._HR_reconstruction = self.networks['De'].forward(flip_HR_map_id, self._HR_featuremap_e)
            self._LR_reconstruction = self.networks['De'].forward(flip_HR_map_id, self._LR_featuremap_e)
            self.losses['Gu_rsc'] = self._mse_loss(self._LR_reconstruction,
                                                   self._HR_reconstruction.detach()) * self._opt.L_cons_sim
            loss_Gu += self.losses['Gu_rsc']

        return loss_Gu

    def update_learning_rate(self):
        for key in self.lr_schedulers:
            self.lr_schedulers[key].step()

    def evaluate(self, input):
        HR_img = input['HR_img'].cuda(non_blocking=True)
        LR_img = input['LR_img'].cuda(non_blocking=True)
        with torch.no_grad():
            _, HR_feature_e, _, _ = self.networks['En_h'].forward(HR_img)
            LR_feature_e, _ = self.networks['En_l'].forward(LR_img)
            HR_beha_logit = self.networks['C_e'].forward(HR_feature_e)
            LR_beha_logit = self.networks['C_e'].forward(LR_feature_e)

            HR_id_logit = self.networks['C_id_adv'].forward(HR_feature_e)
            LR_id_logit = self.networks['C_id_adv'].forward(LR_feature_e)
        return HR_beha_logit, HR_id_logit, LR_beha_logit, LR_id_logit

    def get_current_lr(self):
        return self.optimizers['En_h'].param_groups[0]['lr']

    def get_LR_feature(self, img):
        img = img.cuda()
        with torch.no_grad():
            LR_feature_e, _ = self.networks['En_l'].forward(img)
        return LR_feature_e

    def get_current_errors(self):
        loss_dict = OrderedDict([(key, self.losses[key].item()) for key in self.losses])
        return loss_dict

    def save(self, label):
        # save networks
        for name in self.networks:
             self._save_network(self.networks[name], name, label)

        # save optimizers
        for name in self.optimizers:
            self._save_optimizer(self.optimizers[name], name, label)

    def load(self):
        load_epoch = self._opt.load_epoch
        for name in self.networks:
             self._load_network(self.networks[name], name, load_epoch)

        for name in self.optimizers:
            self._load_optimizer(self.optimizers[name], name, load_epoch)

    def cal_infer_time(self):
        start_time = time.time()
        self._LR_feature_e, self._LR_featuremap_e = self.networks['En_l'].forward(self._LR_img)
        LR_beha_logit = self.networks['C_e'].forward(self._LR_feature_e)
        duration = time.time() - start_time
        print('testing infer time = %.4es'%duration)

    # ------- for generator -------
    def get_current_fake_img(self):
        LR_reconstruction = self.networks['De'].forward(self._HR_featuremap_id.flip(0), self._HR_featuremap_e)
        return self._HR_reconstruction, LR_reconstruction


class LossInequalityRegulation(nn.Module):
    def __init__(self, opt):
        super(LossInequalityRegulation, self).__init__()
        self._zeos = torch.zeros((opt.batch_size, opt.expression_type)).cuda()

    def forward(self, loss_l, loss_h):
        inequality = loss_l - loss_h
        # print(inequality)
        l_lir = torch.max(torch.stack((inequality, self._zeos), dim=0), dim=0)[0]
        # print(llir.sum())
        return l_lir.sum()
