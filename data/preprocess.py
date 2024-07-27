#!/usr/bin/python3
# coding=utf-8
"""
@Author: 梁聪
@last modified: 2022/1/5 10:39
"""
import os
from glob import glob

import pickle
import matplotlib.pyplot as plot
import numpy as np

def collect_features(train_feature_files, test_feature_files, train_out, test_out):
    pass

def glob_features(feature_root):
    train_feat_files = glob(os.path.join(feature_root, 'train', '*.pkl'))
    test_feat_files = glob(os.path.join(feature_root, 'test', '*.pkl'))
    print('train len %d, test len %d' % (len(train_feat_files), len(test_feat_files)))
    return train_feat_files, test_feat_files

def make_img_csvs(expr_dir):
    train_metadata = []
    train_imgs = glob(os.path.join(expr_dir, 'imgs', 'train', '*.png'))
    for img_path in train_imgs:
        img_name = os.path.basename(img_path)
        real_id, _, exp, _ = img_name[:-4].split('_')
        real_id = real_id[2:]
        exp = exp[1:]
        train_metadata.append([os.path.join('train', img_name), exp, real_id, img_name])
    print(len(train_metadata))
    with open(os.path.join(expr_dir, 'train.csv'), 'w') as f:
        f.write('\n'.join([','.join(m) for m in train_metadata]))

    test_metadata = []
    test_imgs = glob(os.path.join(expr_dir, 'imgs', 'test', '*.png'))
    for img_path in test_imgs:
        img_name = os.path.basename(img_path)
        real_id, _, exp, _ = img_name[:-4].split('_')
        real_id = real_id[2:]
        exp = exp[1:]
        test_metadata.append([os.path.join('test', img_name), exp, real_id, img_name])
    print(len(test_metadata))
    with open(os.path.join(expr_dir, 'test.csv'), 'w') as f:
        f.write('\n'.join([','.join(m) for m in test_metadata]))

def draw(expr_dir):
    train_log = open(os.path.join(expr_dir, 'train.log'), 'r').readlines()
    train_log = [l for l in train_log if ',' in l]
    print('train log len', len(train_log))
    # train_history = {'i_epoch':[], 'real_id':[], 'read_exp':[],
    #                  'gen_id':[], 'gen_fake_id':[], 'gen_exp':[]}
    train_history = {}
    for l in train_log:
        ep, real, fake = l.split(',')
        ep = int(ep.split(' ')[1])
        # print(ep)
        real_id, real_exp = real.split(':')[1:]
        real_id = float(real_id.split('-')[-1])
        real_exp = float(real_exp.split('-')[-1])
        # print(real_id, real_exp)
        gen_id, gen_fake_id, gen_exp = fake.split('. t')[0].split(":")[1:]
        gen_id = float(gen_id.split('-')[-1])
        gen_fake_id = float(gen_fake_id.split('-')[-1])
        gen_exp = float(gen_exp.split('-')[-1])
        # print(gen_id, gen_fake_id, gen_exp)
        train_history[ep] = (real_id, real_exp, gen_id, gen_fake_id, gen_exp)
        # train_history['i_epoch'].append(ep)
        # train_history['real_id'].append(real_id)
        # train_history['read_exp'].append(real_exp)
        # train_history['gen_id'].append(gen_id)
        # train_history['gen_fake_id'].append(gen_fake_id)
        # train_history['gen_exp'].append(gen_exp)
    ep, real_id, real_exp, gen_id, gen_fake_id, gen_exp = [],[],[],[],[],[]
    for i in sorted(list(train_history.keys())):
        ep.append(i)
        real_id.append(train_history[i][0])
        real_exp.append(train_history[i][1])
        gen_id.append(train_history[i][2])
        gen_fake_id.append(train_history[i][3])
        gen_exp.append(train_history[i][4])
    plot.figure('train_real', figsize=(20, 5))
    plot.plot(ep, real_id, color='xkcd:blue', label='real_id')
    plot.plot(ep, real_exp, color='xkcd:green', label='real_exp')
    plot.xlabel('epochs')
    plot.ylabel('acc')
    plot.legend(loc='lower right')
    # plot.show()
    plot.savefig(os.path.join(expr_dir, 'train_real.jpg'), bbox_inches='tight')

    plot.figure('train_gen', figsize=(20, 5))
    plot.plot(ep, gen_id, color='xkcd:red', label='gen_id')
    plot.plot(ep, gen_fake_id, color='xkcd:orange', label='gen_fake_id')
    plot.plot(ep, gen_exp, color='xkcd:fuchsia', label='gen_exp')
    plot.xlabel('epochs')
    plot.ylabel('acc')
    plot.legend(loc='lower right')
    # plot.show()
    plot.savefig(os.path.join(expr_dir, 'train_gen.jpg'), bbox_inches='tight')

    """" attaker I """
    attackerI_test = pickle.load(open(os.path.join(expr_dir, 'attackerI_eval_history.pkl'), 'rb'))
    print(list(attackerI_test.keys()))
    ep = list(attackerI_test.keys())
    id_acc, exp_acc = [], []
    for i in ep:
        total_i = attackerI_test[i]['total']
        id_acc.append(attackerI_test[i]['id_acc']/total_i)
        exp_acc.append(attackerI_test[i]['exp_acc']/total_i)

    plot.figure('attackerI', figsize=(12, 4))
    plot.plot(ep, id_acc, color='xkcd:red', label='attackI id, max:%.3f'%max(id_acc))
    plot.plot(ep, exp_acc, color='xkcd:green', label='attackI exp, max:%.3f'%max(exp_acc))
    plot.xlabel('epochs')
    plot.ylabel('acc')
    plot.legend(loc='lower right')
    # plot.show()
    plot.savefig(os.path.join(expr_dir, 'attackerI.jpg'), bbox_inches='tight')

    """" attaker II """
    attackerII_test = pickle.load(open(os.path.join(expr_dir, 'attackerII_eval_history.pkl'), 'rb'))
    print(list(attackerII_test.keys()))
    ep = list(attackerII_test.keys())
    id_acc, exp_acc = [], []
    for i in ep:
        total_i = attackerII_test[i]['total']
        id_acc.append(attackerII_test[i]['id_acc']/total_i)
        exp_acc.append(attackerII_test[i]['exp_acc']/total_i)

    plot.figure('attackerII', figsize=(12, 4))
    plot.plot(ep, id_acc, color='xkcd:red', label='attackII id, max:%.3f'%max(id_acc))
    plot.plot(ep, exp_acc, color='xkcd:green', label='attackII exp, max:%.3f'%max(exp_acc))
    plot.xlabel('epochs')
    plot.ylabel('acc')
    plot.legend(loc='lower right')
    # plot.show()
    plot.savefig(os.path.join(expr_dir, 'attackerII.jpg'), bbox_inches='tight')


    """" attaker III """
    attackerIII_test = pickle.load(open(os.path.join(expr_dir, 'attacker_eval_history.pkl'), 'rb'))
    print(list(attackerIII_test.keys()))
    ep = list(attackerIII_test.keys())
    id_acc, exp_acc = [], []
    for i in ep:
        total_i = attackerIII_test[i]['total']
        id_acc.append(attackerIII_test[i]['id_acc']/total_i)
        exp_acc.append(attackerIII_test[i]['exp_acc']/total_i)

    plot.figure('attackerIII', figsize=(8, 4))
    plot.plot(ep, id_acc, color='xkcd:red', label='attackIII id, max:%.3f'%max(id_acc))
    plot.plot(ep, exp_acc, color='xkcd:green', label='attackIII exp, max:%.3f'%max(exp_acc))
    plot.xlabel('epochs')
    plot.ylabel('acc')
    plot.legend(loc='lower right')
    # plot.show()
    plot.savefig(os.path.join(expr_dir, 'attackerIII.jpg'), bbox_inches='tight')


if __name__ == '__main__':
    # glob_features('checkpoints/PPRL_VGAN_G2/features')
    # make_img_csvs('checkpoints/PPRL_VGAN_D2/')
    draw('checkpoints/PPRL_VGAN_G2/')