"""References:
Guangcan Mai, Kai Cao, Pong C. Yuen and Anil K. Jain. 
"On the Reconstruction of Face Images from Deep Face Templates." 
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2018)
"""

from symbol_nbnet import get_symbol
from scipy.spatial.distance import cosine
from util.util import *
from data.LRPPN_data_loader import LRPPNDataLoader
# import tensorflow as tf
import os,errno
import mxnet as mx
from mxnet.io import DataBatch
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import logging
import time

mx.random.seed(100)
from datetime import datetime

# TensorBorad Flag
UseTensorBoard = False

if UseTensorBoard: 
    from TensorBoardLogging import LogMetricsCallback

def main(args):
    print('**************** preparing logger')
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
    log_file_full_name = '%s_%s.log'%(args.model_save_prefix, stamp)
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    if UseTensorBoard: 
        tb_folder = args.model_save_prefix+'_tblog/train'
        try:
            os.makedirs(tb_folder)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(tb_folder):
                pass
            else:
                raise
        batch_end_tb_callback = LogMetricsCallback(tb_folder,score_store=True)

    print('**************** loading NBNet')
    net = get_symbol(6, net_switch=args.net_switch)     # symbol = pytorch 中网络层，总之是取 NBNet，6是block数量，switch 选类型
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]  # _device
    devsid = [int(i) for i in args.gpus.split(',')]     # 所有可用 gpu
    #pdb.set_trace()
    checkpoint = mx.callback.do_checkpoint(args.model_save_prefix)
    speedmeter = mx.callback.Speedometer(args.batch_size, 50)
    kv = mx.kvstore.create(args.kv_store)   # 键值对
    arg_params = None
    aux_params = None
    if args.retrain:
        print('************ loading NBNet', args.model_load_epoch, 'from '+args.model_load_prefix)
        _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)

    print('**************** loading DCGAN and feature extractor')
    img_gen = img_generator(ctx=devs, batch_size=args.batch_size, 
                        model_prefix=args.face_gen_prefix, load_epoch=args.face_gen_epoch)  # 加载 DCGAN
    if args.LRPPN_path:
        print('************ use LRPPN')
        fea_ext = LRPPN_feature_extractor(args, devsid, args.LRPPN_path, args.LRPPN_img_size, args.is_HR)     # 在所有可用 gpu 上加载 LRPPN_feature_extractor
    else:
        print('************ use FaceNet')
        fea_ext = feature_extractor(devsid)     # 在所有可用 gpu 上加载 feature_extractor
    print('**************** preparing data iter')
    train = RandFaceIter(batch_size=args.batch_size, img_gen=img_gen, fea_ext=fea_ext,
                         max_count=args.dataset_size//args.batch_size,
                         feature_size=16384 if args.LRPPN_path else 128)      # 相当于 pytorch DataLoader
    train = mx.io.PrefetchingIter(train)       # 多线程取数据

    print('**************** load NBNet to GPU(s): '+str(devs))
    model = mx.mod.Module(symbol=net, context=devs)     # 把 NBNet 扔到 _device 上
    print('**************** model.bind')
    model.bind(data_shapes=train.provide_data, label_shapes=None, for_training=True)
    print('**************** model.init_params')
    model.init_params(
        initializer=mx.init.Normal(0.02),
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True
        )
    # return
    print('**************** init optimizer')
    model.init_optimizer(
        optimizer          = 'adam',
        optimizer_params = {
            "learning_rate" : 5e-4,
            "wd" : 0,
            "beta1" : 0.5,
            "lr_scheduler" : mx.lr_scheduler.FactorScheduler(step=5000, factor=0.98)
        })

    logging.info('*************** start with arguments %s', args)
    def cosine_score(label,pred):
        return 1.0-((label-pred)**2.0).mean()*label.shape[1]/2.

    #tpl_score = mx.metric.CustomMetric(feval=cosine_score,name='tpl_score')
    pixel_mae = mx.metric.MAE(name='pix_mae')   # loss
    eval_metrics = CustomMetric()       # 收集所有 loss
    #eval_metrics.add(tpl_score)
    eval_metrics.add(pixel_mae)

    print('**************** start training')
    total_start_time = time.time()
    epoch_start_time = total_start_time
    for epoch in range(args.model_load_epoch if args.retrain else 0, args.n_epoch):#train at most 80 epoches
        print('**************** start epoch %d' % epoch)
        logging.info('*************** start epoch %d' % epoch)
        tic = time.time()
        num_batch = 0

        end_of_batch = False        # batch 循环条件
        train.reset()               # 重置数据集
        data_iter = iter(train)     # 数据迭代器
        pixel_mae.reset()           #
        next_data_batch = next(data_iter)
        if epoch == 0:
            print('data description:', next_data_batch)
        # break
        batch_start_time = time.time()
        while not end_of_batch:
            data_batch = next_data_batch
            num_batch += 1
            # print('data_batch type', type(data_batch))
            # print('data_batch.data type', type(data_batch.data))
            # print('data_batch.data element type', type(data_batch.data[0]))
            model._exec_group.forward(data_batch,is_train=True) # forward
            rec_img = model._exec_group.get_outputs()

            try:# pre fetch  next batch
                next_data_batch = next(data_iter)
                model.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True

            eval_metrics.update(data_batch.label,rec_img)
            img_grad = mx.nd.sign(rec_img[0].as_in_context(mx.cpu()) - data_batch.label[0])
            model._exec_group.backward([img_grad])
            model.update()

            if num_batch % 1000 == 0:
                rep_ori = data_batch.data[0].asnumpy()
                rec_img_tmp = np.clip((rec_img[0].asnumpy().transpose((0,2,3,1))+1.0)/2.0, 0, 1)
                rec_img_tmp = [np.expand_dims(prewhiten(x),axis=0) for x in rec_img_tmp]
                rep_rec = fea_ext(np.concatenate(rec_img_tmp))
                facenet_scores = np.array([cosine(x,y) for x,y in zip(rep_ori, rep_rec)])
                batch_duration = time.time() - batch_start_time
                batch_start_time += batch_duration
                print('*************** training epoch %d 1000 batch use: %dmin:%.1fs' %
                      (epoch, batch_duration//60, batch_duration%60))
                print('*************** training epoch %d batch %d: avg face_net_score %f' %
                      (epoch, num_batch, facenet_scores.mean()))
                logging.info('*************** training epoch %d batch %d: avg face_net_score %f' %
                      (epoch, num_batch, facenet_scores.mean()))

            batch_end_params = BatchEndParam(epoch=epoch,
                nbatch=num_batch,
                eval_metric=eval_metrics,
                locals=locals())
            if UseTensorBoard: 
                batch_end_tb_callback(batch_end_params) 
            speedmeter(batch_end_params)


        arg_params, aux_params = model.get_params()
        checkpoint(epoch,model.symbol,arg_params, aux_params)

        epoch_duration = time.time() - epoch_start_time
        avg_duration = (time.time() - total_start_time)/(epoch+1)
        epoch_start_time = time.time()
        print('epoch %d duration %dmin:%.1fs, avg epoch duration %dmin:%.1fs, estimated left %dh:%.1fmin' %
              (epoch, epoch_duration//60, epoch_duration%60, avg_duration//60, avg_duration%60,
               avg_duration*(79-epoch)//3600, avg_duration*(79-epoch)%3600/60))

def test(args):
    """ test NBNet with LRPPN low-resolution encoder """
    print('**************** loading NBNet')
    net = get_symbol(6, net_switch=args.net_switch)  # symbol = pytorch 中网络层，总之是取 NBNet，6是block数量，switch 选类型
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]  # _device
    devsid = [int(i) for i in args.gpus.split(',')]  # 所有可用 gpu
    print('************ loading NBNet', args.model_load_epoch, 'from ' + args.model_load_prefix)
    _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)

    if args.LRPPN_path:
        print('************ use LRPPN')
        fea_ext = LRPPN_feature_extractor(args, devsid, args.LRPPN_path, args.LRPPN_img_size, args.is_HR)     # 在所有可用 gpu 上加载 LRPPN_feature_extractor
    else:
        print('************ use FaceNet')
        fea_ext = feature_extractor(devsid)     # 在所有可用 gpu 上加载 feature_extractor

    print('**************** get dataloader')
    dataloader = LRPPNDataLoader(args)

    print('**************** load NBNet to GPU(s): ' + str(devs))
    model = mx.mod.Module(symbol=net, context=devs)  # 把 NBNet 扔到 _device 上
    print('**************** model.bind')
    model.bind(data_shapes=[('data', (1, 16384) if args.LRPPN_path else (0, 128))], label_shapes=None, for_training=False)
    print('**************** model.init_params')
    model.init_params(
        initializer=mx.init.Normal(0.02),
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True
    )

    save_img_dir = os.path.join(os.path.dirname(args.model_save_prefix), 'imgs')
    if not os.path.exists(save_img_dir):
        print('makedirs', save_img_dir)
        os.makedirs(save_img_dir, exist_ok=True)
    for test_batch in tqdm(dataloader.load_data()):
        if args.is_HR:
            imgs = test_batch['HR_img'].numpy()
        else:
            imgs = test_batch['LR_img'].numpy()
        # print('imgs shape', imgs.shape, 'dtype', imgs.dtype)
        features = fea_ext(imgs)
        # print('LRPPN out shape', features.shape, 'type', type(features))

        mx_input = DataBatch([mx.nd.array(features)])
        # if not args.LRPPN_path:
        #     print('data_batch type', type(mx_input))
        #     print('data_batch.data type', type(mx_input.data))
        #     print('data_batch.data element type', type(mx_input.data[0]), mx_input.data[0].shape)
        model._exec_group.forward(mx_input, is_train=False)  # forward
        rec_img = model._exec_group.get_outputs()
        # print('NBNet out type', type(rec_img[0]))
        # print('NBNet out shape', rec_img[0].shape)
        rec_img = rec_img[0].asnumpy()
        rec_img = np.clip((rec_img.transpose((0,2,3,1))+1.0)/2.0, 0, 1)*255
        rec_img = rec_img.astype(np.uint8)

        for idx, file_path in enumerate(test_batch['LR_img_path']):
            file_name = os.path.basename(file_path)
            cv2.imwrite(os.path.join(save_img_dir, file_name), imgs[idx])

            file_name = file_name[:-4] + '_fake'+file_name[-4:]
            cv2.imwrite(os.path.join(save_img_dir, file_name), cv2.cvtColor(rec_img[idx],cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    # region cmd args
    parser = argparse.ArgumentParser(description="command for training openface-generator-cnn")
    parser.add_argument('--is_train', action='store_true', help='true for training, default for testing')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model-save-prefix', type=str, default='../model/nbnet/of2img',
                        help='the prefix of the model to save')
    parser.add_argument('--dataset-size', type=int, default=64 * 5000, help='the batch size')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch size')
    parser.add_argument('--n-epoch', type=int, default=80, help='the batch size')
    parser.add_argument('--net-switch', type=str, default='nbnetb', choices=['nbneta', 'nbnetb', 'dcnn'],
                        help='the nbnet architecture to be used')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--model-load-prefix', type=str, default=None, help='the prefix of the model to load')
    parser.add_argument('--model-load-epoch', type=int, default=0, help='indicate the epoch for model-load-prefix')
    parser.add_argument('--face-gen-prefix', type=str, default='../model/dcgan/vgg-160-pt-slG',
                        help='the prefix of the face generation model to load')
    parser.add_argument('--face-gen-epoch', type=int, default=0, help='epoch for loading face generation model')
    parser.add_argument('--LRPPN_path', type=str, default='',
                        help='the path of LRPPN En_l model')
    parser.add_argument('--LRPPN_img_size', type=int, default=128,
                        help='(3, img_size, img_size), the input size of LRPPN En_l model')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    parser.add_argument('--data_dir', type=str, default='', help='LRPPN input dataset root')
    parser.add_argument('--ids_file_suffix', type=str, default='_16.csv', help='LRPPN csv')
    parser.add_argument('--HR_image_size', type=int, default=128, help='input size of LRPPN')
    parser.add_argument('--is_HR', action='store_true', default=False, help='train on En_h')

    args = parser.parse_args()
    args.model_save_prefix += '-'+args.net_switch+'-'+args.face_gen_prefix.split('/')[-1].split('-')[0]
    args.model_load_prefix += '-'+args.net_switch+'-'+args.face_gen_prefix.split('/')[-1].split('-')[0]
    # endregion

    save_dir = os.path.dirname(args.model_save_prefix)
    if not os.path.exists(save_dir):
        print('makedirs', save_dir)
        os.makedirs(save_dir, exist_ok=True)

    if args.is_train:
        main(args)
    else:
        test(args)