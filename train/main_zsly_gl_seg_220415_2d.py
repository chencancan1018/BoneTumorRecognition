import os
# os.environ['CUDA_VISIBLE_DEVICES']='2,3'
# os.environ['RANK']='1'
# os.environ['WORLD_SIZE']='1'
# os.environ['LOCAL_RANK']='4'

import argparse
import datetime
import numpy as np
import time
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy  #, SoftFocalLoss, GateSoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import pandas as pd
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from engine_220415 import train_one_epoch, evaluate, evaluate_eval
from samplers import RASampler
import utils
from args import get_args_parser

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import MRIDataset_2p5D
from cross_padding import random_crop_to_fix_shape
from utils import compute_iou

from loss import SoftFocalLoss

###########################################################################################

def product_boxes(img, fix_shape=(64, 64), iou_thresh=0.3):
    if img.shape[0] == fix_shape[0]:
        box_tmp = [0, 0, fix_shape[0], fix_shape[1]]
        return [box_tmp, box_tmp]
    box_1 = random_crop_to_fix_shape(img, fix_shape=fix_shape)
    # flag_ = True
    # box_2 = random_crop_to_fix_shape(img, fix_shape=fix_shape)
    while 1:
        # print('debug_show2')
        # print('=====debug', img.shape)
        box_2 = random_crop_to_fix_shape(img, fix_shape=fix_shape)
        iou_ = compute_iou(box_1, box_2)
        if iou_ < iou_thresh:
            # flag_=False
            return [box_1, box_2]

def product_boxes_valid(img, fix_shape=(64, 64), iou_thresh=0.3):  # 0.3
    # area_ratio_ = (img.shape[0]*img.shape[1])/(fix_shape[0]*fix_shape[1]+1e-6)
    if img.shape[0] <= fix_shape[0]:
        box_tmp = [0, 0, fix_shape[0], fix_shape[1]]
        return [box_tmp, box_tmp]
    box_1 = crop_center_v2(img, fix_shape[0], fix_shape[1])
    # box_2 = [0, 0, fix_shape[0], fix_shape[1]]
    # return [box_1, box_2]
    # print('debug into', img.shape)
    while 1:
        # print('debug_show3')
        box_2 = random_crop_to_fix_shape(img, fix_shape=fix_shape)
        iou_ = compute_iou(box_1, box_2)
        if iou_ < iou_thresh:
            # print('debug out')
            # flag_=False
            return [box_1, box_2]

def lumTrans(img):
    lungwin = np.array([-1024., 400.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg

def window_transfer(data, window_center, window_width):
    window_low = window_center - window_width // 2.
    new_data = (data - window_low) / float(window_width)
    new_data[new_data < 0] = 0
    new_data[new_data > 1] = 1
    new_data = (new_data * 255).astype(np.uint8)
    return new_data.copy()

def crop_center_v2(img, croph, cropw):
    height, width = img.shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return [starth, startw, starth + croph, startw + cropw]

# @jit
# def aug_pipe(img, has_imgaug=True, random_rotate=True, random_lightness=True, random_transpose=True, im_pad_val=0):
#     h_, w_ = img.shape
#     img_ = np.zeros((1, h_, w_), dtype=np.float32)
#     img_[0] = img
#     if has_imgaug:
#         img_ = img_.transpose((2, 1, 0))
#         img_ = aug_process(img_, im_pad_val=im_pad_val)
#         img_ = img_.transpose((2, 1, 0))
#
#     return img_[0]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def get_split_deterministic(all_keys, fold=0, num_splits=4, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """

    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys

def func_1130_1(df_tmp, ids_list,ind_name='pid', used_ind='img_npy_path'):
    img_dir_path_list = []
#    mri_series_list = []
    for i in range(len(ids_list)):
        case_pid_tmp_ = ids_list[i]
        img_dir_path_list_tmp = df_tmp[df_tmp[ind_name] == case_pid_tmp_][used_ind].tolist()
        img_dir_path_list += img_dir_path_list_tmp
#        mri_series_list_tmp = df_tmp[df_tmp[ind_name] == case_pid_tmp_]['mri_series'].tolist()
#        mri_series_list += mri_series_list_tmp
#    if test:
#        print(img_dir_path_list, mri_series_list)
#        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
#        return img_dir_path_list, mri_series_list
#    else:
    return img_dir_path_list

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    # Debug mode.
    if args.debug:
        import debugpy
        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    ################################################################
    csvPath = args.data_path
    train_fold_idx = args.fold
    tfold = args.tfold
    has_bootstrap = args.has_bootstrap
    fix_data_shape = args.crop_size
    print('*****: ', fix_data_shape)
    ind_name = 'pid'
    cls_weight_list = [1, 1, 1, 1, 1, 1, 1]
    if train_fold_idx == -1:
        # df_tmp = pd.read_excel(csvPath, sheet_name='Sheet1')
        df_tmp = pd.read_csv(csvPath)
        train_idx = np.array(df_tmp[df_tmp['dataset'] != 'test'][ind_name].unique())
        if has_bootstrap:
            train_idx = list(train_idx) * has_bootstrap
            train_idx = np.array(train_idx)
        valid_idx = np.array(df_tmp[df_tmp['dataset'] == 'test'][ind_name].unique())

        train_sids = func_1130_1(df_tmp, train_idx, used_ind='img_npy_path')
        np.random.shuffle(train_sids)
        valid_sids = func_1130_1(df_tmp, valid_idx, used_ind='img_npy_path')
        np.random.shuffle(valid_sids)
        # test_sids = func_1130_1(df_tmp, test_idx)
        # np.random.shuffle(test_sids)
    else:
        if tfold < 0:
            np.random.seed(12345)
            df_tmp = pd.read_csv(csvPath)
            df_tmp = df_tmp[df_tmp['mri_series'].map(lambda x: x in args.series)]
            print(args.series, df_tmp.shape, type(args.series))
            train_idx_list = []
            valid_idx_list = []
            label_name_list = df_tmp['label'].unique().tolist()
            num_cls_tmp = len(label_name_list)
            # cls_weight_list = [1, 1, 1, 3, 1, 1, 1]
            for i in range(num_cls_tmp):
                label_name_ = label_name_list[i]
                patients0 = np.array(df_tmp[(df_tmp['label'] == label_name_) & (df_tmp['dataset'] != 'test')][ind_name].unique())
                train_idx0, valid_idx0 = get_split_deterministic(patients0, fold=train_fold_idx, num_splits=5)
                if has_bootstrap:
                    train_idx0 = list(train_idx0) * int(has_bootstrap * cls_weight_list[i])
                    train_idx0 = np.array(train_idx0)
                train_idx_list.append(train_idx0)
                valid_idx_list.append(valid_idx0)
            train_idx = np.concatenate(train_idx_list, axis=0)
            np.random.shuffle(train_idx)
            valid_idx = np.concatenate(valid_idx_list, axis=0)
            np.random.shuffle(valid_idx)
            test_idx = np.array(df_tmp[df_tmp['dataset'] == 'test'][ind_name].unique())

            train_sids = func_1130_1(df_tmp, train_idx, used_ind='img_path')
            np.random.shuffle(train_sids)
            valid_sids = func_1130_1(df_tmp, valid_idx, used_ind='img_path')
            np.random.shuffle(valid_sids)
            test_sids  = func_1130_1(df_tmp, test_idx, used_ind='img_path')
            np.random.shuffle(test_sids)
        elif tfold == 100:
            df_tmp = pd.read_csv(csvPath)
            train_idx = []
            valid_idx = []
            test_idx = np.array(df_tmp[ind_name].unique())
            train_sids = []
            valid_sids = []
            test_sids = func_1130_1(df_tmp, test_idx, used_ind='img_path')
        else:
            np.random.seed(12345)
            df_tmp = pd.read_csv(csvPath)
            # df_tmp = df_tmp[df_tmp['Modality_Index']==1]
            train_idx_list=[]
            valid_idx_list=[]
            test_idx_list=[]
            label_name_list = df_tmp['label'].unique().tolist()
            num_cls_tmp = len(label_name_list)
            # cls_weight_list = [1,1,1,1,1,1,1]
            for i in range(num_cls_tmp):
                label_name_ = label_name_list[i]
                patients0 = np.array(df_tmp[(df_tmp['label'] == label_name_)][ind_name].unique())
                train_idx00, test_idx0 = get_split_deterministic(patients0, fold=tfold, num_splits=3)
                train_idx0, valid_idx0 = get_split_deterministic(train_idx00, fold=train_fold_idx, num_splits=4)
                if has_bootstrap:
                    train_idx0 = list(train_idx0) * int(has_bootstrap*cls_weight_list[i])
                    train_idx0 = np.array(train_idx0)
                train_idx_list.append(train_idx0)
                valid_idx_list.append(valid_idx0)
                test_idx_list.append(test_idx0)
            train_idx = np.concatenate(train_idx_list, axis=0)
            np.random.shuffle(train_idx)
            valid_idx = np.concatenate(valid_idx_list, axis=0)
            np.random.shuffle(valid_idx)
            test_idx = np.concatenate(test_idx_list, axis=0)
            np.random.shuffle(test_idx)

            train_sids = func_1130_1(df_tmp, train_idx, used_ind='img_npy_path')
            np.random.shuffle(train_sids)
            valid_sids = func_1130_1(df_tmp, valid_idx, used_ind='img_npy_path')
            np.random.shuffle(valid_sids)
            test_sids = func_1130_1(df_tmp, test_idx, used_ind='img_npy_path')
            np.random.shuffle(test_sids)

    print('==== num of training pids: ', len(train_idx))
    print('==== num of validation pids: ', len(valid_idx))
    print('==== num of testing pids: ', len(test_idx))
    print('@@@@==== num of training sids: ', len(train_sids))
    print('@@@@==== num of validation sids: ', len(valid_sids))
    print('@@@@==== num of testing sids: ', len(test_sids))
    ################################################################
    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    # dataset_val, _ = build_dataset(is_train=False, args=args)


    valid_mode = 'valid'
    index_list_tmp_ = valid_idx
    if args.eval:
        valid_mode = 'eval'
        if args.pred_state == 'train':
            index_list_tmp_ = train_idx
        elif args.pred_state == 'valid':
            index_list_tmp_ = valid_idx
        else:
            index_list_tmp_ = test_idx
        if args.testing_aug_num:
            index_list_tmp_ = list(index_list_tmp_) * int(args.testing_aug_num)

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=45, p=0.5)
        ]
    )
    dataset_train = MRIDataset_2p5D(index_list=train_idx,
                                    data_shape=(args.in_chans, fix_data_shape, fix_data_shape),
                                    label_name='label',
                                    idx_name=ind_name,
                                    mode='train',
                                    csv_path=csvPath,
                                    extend_num=args.extend_num,
                                    transform=train_transform,
                                    fold = tfold,
                                    mri_list=args.series
                                    )
    dataset_val = MRIDataset_2p5D(index_list=index_list_tmp_,
                                  data_shape=(args.in_chans, fix_data_shape, fix_data_shape),
                                  label_name='label',
                                  idx_name=ind_name,
                                  mode=valid_mode,
                                  csv_path=csvPath,
                                  extend_num=args.extend_num,
                                  fold=tfold,
                                  mri_list=args.series)


    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(1. * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    print(f"Creating model: {args.model}")
    # model = DenseUNet3d(args.num_classes)

    model = smp.Unet(              ### PSPNet, Unet, DeepLabV3Plus, UnetPlusPlus, FPN, MAnet, Linknet
        encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet", 
        in_channels=3,  
        classes=args.num_classes,  
    )
    # model = smp.UnetPlusPlus(
    #     encoder_name="timm-efficientnet-b8",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=args.num_classes,  # model output channels (number of classes in your dataset)
    # )
    # model = create_model(
    #     args.model,
    #     pretrained=True,
    #     num_classes=args.num_classes,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=args.drop_block,
    #     in_chans=args.in_chans,
    #     img_size=fix_data_shape,
    #     final_drop=args.final_drop,
    #     mode='eval' if args.eval else 'train',
    #     **eval(args.model_kwargs))
    # print(model)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters)
    print('lr: ', args.lr)
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    # criterion = SoftFocalLoss(6, w_list=[0.1, 1, 1, 1, 1, 1],smoothing=args.smoothing)
    # criterion = MultiClassDiceLoss()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        # criterion = GateSoftTargetCrossEntropy()
        # criterion = SoftFocalLoss(3, w_list=[1, 1, 1], smoothing=args.smoothing, is_mixup=True)

    elif args.smoothing:
        # criterion = SoftFocalLoss(3, w_list=[1, 2.3, 2.3], smoothing=args.smoothing)
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        # criterion = SoftFocalLoss(6, w_list=[0.1, 1, 1, 1, 1, 1], smoothing=args.smoothing)
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            raise NotImplementedError
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    if args.eval:
        test_stats = evaluate_eval(data_loader_val, model, device, disable_amp=args.disable_amp,
                                   num_classes=args.num_classes,
                                   resume=args.resume, fold=args.fold, tfold=args.tfold, testing_aug_num=args.testing_aug_num, pred_state=args.pred_state,output_dir = args.output_dir)
        print(f"mDice of the network on the {len(dataset_val)} test images: {1-test_stats['loss']:.1f}%")
        if args.output_dir and utils.is_main_process():
            with (output_dir / "test_log.txt").open("a") as f:
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'n_parameters': n_parameters}
                f.write(json.dumps(log_stats) + "\n")
        return

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    # Initial checkpoint saving.
    if args.output_dir:
        checkpoint_paths = [output_dir / 'checkpoints/checkpoint.pth']
        # print(checkpoint_paths)
        if not os.path.exists(args.output_dir + '/checkpoints'):
            os.makedirs(args.output_dir + '/checkpoints', exist_ok=True)
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': -1,  # Note: -1 means initial checkpoint.
                'model_ema': get_state_dict(model_ema),
                'args': args,
            }, checkpoint_path)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
                                      args.clip_grad, model_ema, mixup_fn, disable_amp=args.disable_amp)

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoints/checkpoint.pth']
            if epoch % args.save_freq == args.save_freq - 1:
                checkpoint_paths.append(output_dir / f'checkpoints/checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'args': args,
                }, checkpoint_path)

        if epoch % args.save_freq == args.save_freq - 1:
            test_stats = evaluate(data_loader_val, model, device, disable_amp=args.disable_amp)
            print(f"mDice of the network on the {len(dataset_val)} test images: {1-test_stats['loss']:.1f}%")
            max_accuracy = max(max_accuracy, 1-test_stats["loss"])
            print(f'Max mDice: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'mean_dice': max_accuracy,
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
