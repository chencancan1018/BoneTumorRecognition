# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import numpy as np
import pandas as pd

# import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from cross_padding import random_crop_to_fix_shape, pading_fix
import albumentations as A
import SimpleITK as sitk
import random

def read_series(img_path):

    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(img_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.ProcessObject_SetGlobalWarningDisplay(True)

    return image

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    # 有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)  # 限定范围numpy.clip(a, a_min, a_max, out=None)

    # 除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9  # 黑色背景区域
        return tmp


def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)

    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    # print (b)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = random.randint(1, 10) / 10

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def source_to_target_freq(src_img, amp_trg, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img  # .cpu().numpy()
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)

def check_nan(array, position):
    array_sum = np.sum(array)
    array_has_nan = np.isnan(array_sum)
    if array_has_nan:
        print(position, array_has_nan)

class MRIDataset_2p5D(Dataset):
    def __init__(self, index_list, data_shape=(1, 64, 64), label_name='label', idx_name='pid', mode='train',
                 csv_path='', im_pad_val=0, extend_num=0, transform=None, fold=0, mri_list = ['t1c','t2','dwi','t1']):
        super(MRIDataset_2p5D, self).__init__()
        self.mode = mode
        self.idx_name = idx_name
        self.data_shape = data_shape
        self.label_name = label_name
        self.index_list = index_list
        self.im_pad_val = im_pad_val
        self.extend_num = extend_num
        self.len = len(index_list)
        self.all_df = pd.read_csv(csv_path)
        self.transform = transform
        self.fold = fold
        self.mri_list = mri_list

        print('=== mode:' + self.mode)
        print('=== num of samples: ', self.len)

    def sample_generator(self, img_path, mask_path, w_l, h_l, c_l, w_bias, h_bias, c_bias, boxes_list = None, c_ind_list_ = None):

        try:
            img = read_series(img_path)
            img = sitk.GetArrayFromImage(img)
            img = normalize(img)
        except:
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img)
            img = normalize(img)
        
        if boxes_list is None:
            mask = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask)
        else:
            mask = np.zeros(img.shape, dtype=int)
        check_nan(img, 'original image: ')
        check_nan(mask, 'original image: ')

        assert img.shape==mask.shape, 'error of different shape between img and mask'
        cc,hh,ww = img.shape

        if self.mode=='train' or self.mode=='valid':
            if boxes_list is None:
                if (hh < self.data_shape[1]) or (ww < self.data_shape[1]):
                    img, mask = pading_fix(img, mask, is_padding=True, mode='train', im_pad_val=-9, pad_shape=None)

                c_pos_ind_list = [i+c_l for i in range(c_bias) if (i+c_l>0) and (i+c_l<cc-1)]
                c_neg_ind_list = [i for i in range(1,c_l) if (i>0) and (i<cc-1)] + [i for i in range(c_l+c_bias,cc-1) if (i>0) and (i<cc-1)]
                np.random.shuffle(c_pos_ind_list)
                np.random.shuffle(c_neg_ind_list)
                if len(c_neg_ind_list)<1:
                    c_neg_ind_list = c_pos_ind_list
                c_pos_ind_list = c_pos_ind_list[:int(self.data_shape[0])]
                num_neg_ = self.data_shape[0] - len(c_pos_ind_list)
                # num_neg_ = int(self.data_shape[0]*0.5)
                if len(c_neg_ind_list)<num_neg_:
                    num_ = int(num_neg_/(len(c_neg_ind_list)+1e-6))+1
                    c_neg_ind_list = c_neg_ind_list*num_
                    c_neg_ind_list = c_neg_ind_list[:num_neg_]
                else:
                    c_neg_ind_list = c_neg_ind_list[:num_neg_]

                c_ind_list = c_pos_ind_list+c_neg_ind_list
                np.random.shuffle(c_ind_list)
                img_list = []
                mask_list = []
                boxes = []
                for i in c_ind_list:
                    img_tmp = img[i-1:i+2]
                    if self.mode=='train':
                        img_tmp = img_tmp + np.random.uniform(-0.1,0.1)
                    mask_tmp = mask[i:i + 1]

                    box = random_crop_to_fix_shape(img, w_l, h_l, c_l, w_bias, h_bias, c_bias, extend_num=self.extend_num, fix_shape=self.data_shape)
                    boxes.append(box)
                    img_tmp = img_tmp[:, box[1]:box[4], box[2]:box[5]]
                    mask_tmp = mask_tmp[:, box[1]:box[4], box[2]:box[5]]

                    # if self.transform is not None:
                    #     img_tmp = np.transpose(img_tmp, axes=[2, 1, 0])
                    #     mask_tmp = np.transpose(mask_tmp, axes=[2, 1, 0])
                    #     transformed = self.transform(image=img_tmp, mask=mask_tmp)
                    #     img_tmp = transformed['image']
                    #     mask_tmp = transformed['mask']
                    #     img_tmp = np.transpose(img_tmp, axes=[2, 1, 0])
                    #     mask_tmp = np.transpose(mask_tmp, axes=[2, 1, 0])

                    img_tmp = img_tmp[np.newaxis, :, :, :]

                    if img_tmp.shape[1]<3:
                        print(img_tmp.shape, c_ind_list)
                    img_list.append(img_tmp)
                    mask_list.append(mask_tmp)
            else:
                img_list = []
                mask_list = []
                count = 0
                for i in c_ind_list_:
                    img_tmp = img[i - 1:i + 2]
                    if self.mode == 'train':
                        img_tmp = img_tmp + np.random.uniform(-0.1, 0.1)
                    mask_tmp = mask[i:i + 1]

                    box = boxes_list[count]
                    count += 1
                    img_tmp = img_tmp[:, box[1]:box[4], box[2]:box[5]]
                    mask_tmp = mask_tmp[:, box[1]:box[4], box[2]:box[5]]

                    # if self.transform is not None:
                    #     img_tmp = np.transpose(img_tmp, axes=[2, 1, 0])
                    #     mask_tmp = np.transpose(mask_tmp, axes=[2, 1, 0])
                    #     transformed = self.transform(image=img_tmp, mask=mask_tmp)
                    #     img_tmp = transformed['image']
                    #     mask_tmp = transformed['mask']
                    #     img_tmp = np.transpose(img_tmp, axes=[2, 1, 0])
                    #     mask_tmp = np.transpose(mask_tmp, axes=[2, 1, 0])

                    img_tmp = img_tmp[np.newaxis, :, :, :]
                    if img_tmp.shape[1] < 3:
                        print(img_tmp.shape, c_ind_list_)
                    img_list.append(img_tmp)
                    mask_list.append(mask_tmp)
        else:
            img_list = []
            mask_list = []
            if cc >= 200:
                img = img[int(cc/2-60) : int(cc/2+60)]
                mask = mask[int(cc/2-60):int(cc/2+60)]
                cc = img.shape[0]
            if (hh >= 1000) or (ww >= 1000):
                img = img[:,int(hh/2-256):int(hh/2+256),int(ww/2-256):int(ww/2+256)]
                mask = mask[:, int(hh/2-256):int(hh/2+256),int(ww/2-256):int(ww/2+256)]
                cc, hh, ww = img.shape
            for i in range(1,cc-1):
                img_tmp = img[i - 1:i + 2]
                mask_tmp = mask[i:i + 1]
                if img_tmp.shape[1]!=img_tmp.shape[2]:
                    img_tmp, mask_tmp = pading_fix(img_tmp, mask_tmp, is_padding=True, mode='test', im_pad_val=-1, pad_shape=None)
                if img_tmp.shape[1] % 32 != 0:
                    img_tmp, mask_tmp = pading_fix(img_tmp, mask_tmp, is_padding=True, mode='test', im_pad_val=-1, pad_shape=32 * (img_tmp.shape[1]//32 + 1))
    
                img_tmp = img_tmp[np.newaxis, :, :, :]
                img_list.append(img_tmp)
                mask_list.append(mask_tmp)

        img = np.concatenate(img_list).astype('float32')
        mask = np.concatenate(mask_list).astype('int8')
        check_nan(img, 'after transform image: ')
        check_nan(mask, 'after transform image: ')
       #  print('input',img.shape, mask.shape)
        assert img.shape[0]==mask.shape[0],'error in generate sample'


        if boxes_list is None:
            return img, mask, boxes, c_ind_list
        else:
            return img


    def __getitem__(self, index):
        pid = self.index_list[index]
        name = 'img_path'
        img_dir_path_t1c = np.array(self.all_df[(self.all_df['pid'] == pid) & (self.all_df['mri_series'] == 't1c')]["img_path"])[0]
        mask_dir_path_t1c = np.array(self.all_df[(self.all_df['pid'] == pid) & (self.all_df['mri_series'] == 't1c')]["mask_path"])[0]

        w_l = np.array(self.all_df[self.all_df[name] == img_dir_path_t1c]['w_l'])[0]
        h_l = np.array(self.all_df[self.all_df[name] == img_dir_path_t1c]['h_l'])[0]
        c_l = np.array(self.all_df[self.all_df[name] == img_dir_path_t1c]['c_l'])[0]
        w_bias = np.array(self.all_df[self.all_df[name] == img_dir_path_t1c]['w_bias'])[0]
        h_bias = np.array(self.all_df[self.all_df[name] == img_dir_path_t1c]['h_bias'])[0]
        c_bias = np.array(self.all_df[self.all_df[name] == img_dir_path_t1c]['c_bias'])[0]

        random_mri = random.choice(self.mri_list.remove('t1c'))
        pair_img_dir_pth = self.all_df[(self.all_df['pid'] == pid) & (self.all_df['mri_series']==random_mri)][name].tolist()[0]

        img, mask, boxes, c_ind_list = self.sample_generator(img_dir_path_t1c, mask_dir_path_t1c, w_l, h_l, c_l, w_bias, h_bias, c_bias)

        new_img = self.sample_generator(pair_img_dir_pth, mask_dir_path_t1c, w_l, h_l, c_l,w_bias, h_bias, c_bias, boxes_list=boxes, c_ind_list_=c_ind_list)





        if self.mode == 'eval':
            return img, mask, img_dir_path_, mri_series, random_mri, pid
        return img, img_f, mask

    def __len__(self):
        return self.len

###########################################################################################################################
class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder
#####################################################################################################################

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []  # Test-time transformations.
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
