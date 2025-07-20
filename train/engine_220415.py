# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import time
import cv2
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from skimage import exposure
import SimpleITK as sitk
import warnings
from scipy.spatial.distance import directed_hausdorff
warnings.filterwarnings("ignore")
# from sklearn import metrics
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import KFold

def contour_and_draw(image, label_map, n_class=2, shape=(512,512)):
    # image should be (512,512,3), label_map should be (512,512)
    color_lst = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # 'white','red', 'green', 'blue', 'yellow', 'lightblue'
    all_contours = []
    for c_id in range(1, n_class):
        one_channel = np.zeros(shape, dtype=np.uint8)
        one_channel[label_map == c_id] = 1
        contours, hierarchy = cv2.findContours(one_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
        cv2.drawContours(image, contours, -1, color_lst[c_id], 2)
    return image, all_contours

def plot_predict(img_, mask, pred, pid, j, title, save_path):
#    print(img_.shape, mask.shape, pred.shape)
    img_ = exposure.equalize_hist(img_)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_, cmap='gray')
    plt.title('{} Image'.format(title))
    plt.axis('off')

    img_ = np.repeat(img_[:, :, np.newaxis], axis=2, repeats=3).copy()
    img_pred = img_.copy()

    img_mask, _ = contour_and_draw(img_, mask, shape=(img_.shape[0], img_.shape[1]))
    plt.subplot(1, 3, 2)
    plt.imshow(img_mask)
    plt.title('Image with Mask')
    plt.axis('off')

    img_pred, _ = contour_and_draw(img_pred, pred, shape=(img_.shape[0], img_.shape[1]))
    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    plt.title('Image with Predict')
    plt.axis('off')

    dd = os.path.join(save_path,'predict_png')
    os.makedirs(dd, exist_ok=True)

    dir = os.path.join(dd,title, str(pid))
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, '{}.png'.format(j)), dpi=300)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        targets = torch.gt(targets,0)+0
        inputs = F.softmax(inputs, dim=1)
        inputs = torch.argmax(inputs,dim=1)
        inputs = torch.gt(inputs,0)+0

        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice

class BinaryDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, num_classes=6.):
        super(BinaryDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.softmax(inputs, dim=1)
        # inputs = torch.argmax(inputs,dim=1)

        # flatten label and prediction tensors
        inputs = inputs.flatten()
        inputs = torch.gt(inputs, 1./self.num_classes)+0
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice
        # return 1 - dice

class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        # self.kwargs = kwargs

    def forward(self, input, target):
        """
            input tesor of shape = (N, C, H, W)
            target tensor of shape = (N, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        N = input.size(0)
        C = input.size(1)
        target = target.view(-1, 1)
        class_mask = input.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        class_mask.scatter_(1, target.data.long(), 1.)

        assert input.shape == class_mask.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss(num_classes=C)
        total_loss = 0

        # 归一化输出
        logits = F.softmax(input, dim=1)

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(1, C):
            dice_loss = binaryDiceLoss(logits[:, i], class_mask[:, i])
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / C


#####################################
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    disable_amp: bool = False):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True) #2,1,32,224,224
        targets = targets.to(device, non_blocking=True) #2,1,32,224,224
        print('input: ', samples.size(), targets.size())
        B,num_ins,C,H,W = samples.shape
        samples = torch.reshape(samples,(B*num_ins,C,H,W))
        targets = torch.reshape(targets,(B*num_ins,H,W))
        targets = targets.long()
        # targets = torch.flatten(targets)
        # targets = torch.reshape(targets,(-1,1))
        
        # print('debug33: ',samples.shape)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if disable_amp:
            # Disable AMP and try to solve the NaN issue. 
            # Ref: https://github.com/facebookresearch/deit/issues/29
            outputs = model(samples)
#            print(outputs.size(), targets.size())
            loss = criterion(outputs, targets)
        else:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                print('output: ', outputs.size(), targets.size())
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if disable_amp:
            loss.backward()
            optimizer.step()
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, disable_amp):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    # dice_score = MultiClassDiceLoss()
    # binary_dice_score = DiceLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        B,num_ins,C,H,W = images.shape
        images = torch.reshape(images,(B*num_ins,C,H,W))
        target = torch.reshape(target,(B*num_ins,H,W))
        target = target.long()

        # compute output
        if disable_amp:
            output = model(images)
            loss = criterion(output, target)

        else:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
       
        # acc1, acc5 = accuracy(output, target, topk=(1,3))

        # batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('*loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_eval(data_loader, model, device, disable_amp, num_classes, resume, fold, tfold, testing_aug_num, pred_state, output_dir):
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    res_dict = {}
    batch_i = 0
    # if not os.path.exists('img_show'):
    #     os.mkdir('img_show')

    df_eval = []
    img_info = []
    tp_list = 0
    tn_list = 0
    fp_list = 0
    fn_list = 0
    
    start_time = time.time()
    for images, target, img_dir_path_, mri_series, label, pid in metric_logger.log_every(data_loader, 10, header):
        mri_series = mri_series[0]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target.long()
        B,num_ins,C,H,W = images.shape
        images = torch.reshape(images,(B*num_ins,C,H,W))
        target = torch.reshape(target,(B*num_ins,H,W))

        # compute output
        if disable_amp:
            output = model(images)
            loss = criterion(output, target)

        else:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

        metric_logger.update(loss=loss.item())

        output = torch.sigmoid(output)
        # output = torch.argmax(output, dim=1)
        output = torch.reshape(output, (B,1,num_ins,H,W))
        target = torch.reshape(target, (B,1,num_ins,H,W))
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        tp, fp, fn, tn = smp.metrics.get_stats(output.long(), target.long(),mode='binary')
#        print(tp, fp, fn, tn)
#        exit()
        tp_np = tp.detach().cpu().numpy()[0][0]
        fp_np = fp.detach().cpu().numpy()[0][0]
        fn_np = fn.detach().cpu().numpy()[0][0]
        tn_np = tn.detach().cpu().numpy()[0][0]
        tp_list += tp_np
        fp_list += fp_np
        fn_list += fn_np
        tn_list += tn_np

        dices = 2 * tp_np / (fp_np + 2 * tp_np + fn_np + 1e-8)
        ious = tp_np / (tp_np + fp_np + fn_np + 1e-8)

        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metric_logger.meters['IOU'].update(iou_score.item(), n=B)
        
        target = target.detach().cpu().numpy()[0]
        output = output.detach().cpu().numpy()[0]
#        img_ori = img_ori.detach().cpu().numpy()
#        img_ori = img_ori[:,1:-1,:,:]
         
        assert target.shape == output.shape

#        print('output_shape: ', output.shape)
#        print('image_shape: ', img_ori[0].shape)

        pid_ = pid.detach().cpu().numpy()[0]
#        print(pid_)
#        print(pid_)
        dd = os.path.join(os.getcwd(), output_dir,'predict_npy',mri_series)
        os.makedirs(dd, exist_ok=True)
        predict_npy_path = os.path.join(dd, str(pid_)+'.npy')
        output_padded = np.pad(output[0], [(1, 1), (0, 0), (0, 0)], mode='constant', constant_values=0)
        np.save(predict_npy_path, output_padded)
        
        h_list = []
        for k in range(output[0].shape[0]):
            hausdorff_ = directed_hausdorff(output[0][k], target[0][k])
            h_list.append(hausdorff_)
        hausdorff = np.mean(h_list)

        mask_array = output_padded
        c, h, w = mask_array.shape
        non_zero = np.nonzero(mask_array)
        if not np.any(non_zero):
            print(pid_)
            c_l, h_l, w_l = 0, 0, 0
            c_bias, h_bias, w_bias = c-1, h-1, w-1
        else:
            c_l, h_l, w_l = np.min(non_zero[0]), np.min(non_zero[1]), np.min(non_zero[2])
            c_bias, h_bias, w_bias = np.max(non_zero[0])-c_l, np.max(non_zero[1])-h_l, np.max(non_zero[2])-w_l
        label = label.detach().cpu().numpy()[0]
#        print('label: ', label)
        img_info.append([pid_, mri_series,'dev', img_dir_path_[0],  predict_npy_path, c, h, w, c_l, h_l, w_l, c_bias, h_bias, w_bias, label])
        # dir_ = os.path.join('img_show', pid_)
        # if not os.path.exists(dir_):
        #     os.mkdir(dir_)

        # output_padded = np.pad(output[0], [(1, 1), (0,0),(0,0)], mode='constant', constant_values=0)
        # output_pred = sitk.GetImageFromArray(output_padded.astype('uint8'))
        # output_pred.CopyInformation(itk_img)
        # ddd = '/media/tx-deepocean/Data/Infervision/zxly_gl/seg_2.5d/predict_nii'
        # os.makedirs(ddd, exist_ok=True)
        # sitk.WriteImage(output_pred, os.path.join(ddd, pid_ + '.nii.gz'))

        df_eval.append([pid_,mri_series, dices, ious, iou_score.item(), tp_np, fp_np, fn_np, tn_np, hausdorff])

#        for i in range(B):
#            for j in range(num_ins):
#                ori_img = img_ori[i,j,:,:]
#                slice_gt = target[i,j,:,:]
#                slice_pred = output[i,j,:,:]
#                plot_predict(ori_img, slice_gt, slice_pred, pid_, j, mri_series)
        
        # res_dict[batch_i] = [target.cpu().numpy(), output.cpu().numpy(), img_dir_path_]
        del images
        batch_i += 1
    print('total time: ', time.time() - start_time)
    print('* IOU@1 {iou.global_avg:.3f} loss {losses.global_avg:.3f}'.format(iou=metric_logger.IOU, losses=metric_logger.loss))

    mdices = 2 * tp_list / (fp_list + 2 * tp_list + fn_list + 1e-8)
    mious = tp_list / (tp_list + fp_list + fn_list + 1e-8)
#    dff_eval = [[mdices, mious]]
#    m_names = ['mDice', 'mIou']
    dff = pd.DataFrame({'mDice':[mdices], 'mIOU':[mious]})
    dff.to_csv(os.path.join(output_dir,'mean_predict_metrics.csv'), index=False)

    column_names = ['pid','mri_series', 'DICE', 'IOU', 'mean_iou', 'tp','fp','fn','tn', 'hausdorff']
    df_eval_df = pd.DataFrame(df_eval, columns=column_names)
    df_eval_df.to_csv(os.path.join(output_dir, 'predict_metrics.csv'), index=False)
    
    img_info_df = pd.DataFrame(img_info, columns=['pid', 'mri_series','dataset','img_npy_path','mask_npy_path', 'c', 'h', 'w', 'c_l', 'h_l', 'w_l', 'c_bias', 'h_bias', 'w_bias', 'label'])
    img_info_df.to_csv(os.path.join(output_dir, 'predict_info.csv'), index=False)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
