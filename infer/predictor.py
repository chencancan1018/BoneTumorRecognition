import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy.ndimage.interpolation import zoom
from timm.models import create_model
import torch.nn.functional as F

class SegPredictor:

    def __init__(self, gpu, model_pth):
        self.gpu = gpu
        self.model_pth = model_pth
        self.load_model()

    def load_model(self):
        self.net = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
        checkpoint = torch.load(self.model_pth, map_location='cpu')
        self.net.load_state_dict(checkpoint['model'])
        self.net.eval()
        self.net.cuda(self.gpu)

    def forward(self, img_array):
        with torch.no_grad(), torch.cuda.device(self.gpu):
            img_tensor = torch.from_numpy(img_array).cuda().float().unsqueeze(0)

            pred = self.net(img_tensor)
            pred = torch.sigmoid(pred)
            pred[pred>=0.5]=1
            pred[pred<0.5]=0
            pred_slice = pred.squeeze().detach().cpu().numpy()
        return pred_slice

    def predict(self, img_array):
        resize = False
        img_array = self.normalize(img_array)
        c, h, w = img_array.shape
        if h != w:
            resize = True
            edge = np.max([h, w])
            img_array = zoom(img_array, [1, edge/h, edge/w], mode='nearest')
        if img_array.shape[1] % 32 != 0:
            resize = True
            edge = 32 * (img_array.shape[1] // 32 + 1)
            img_array = zoom(img_array, [1, edge/img_array.shape[1], edge/img_array.shape[1]], mode='nearest')

        img_array = np.pad(img_array, [(1, 1), (0, 0), (0, 0)], mode='constant', constant_values=0)
        cc, hh, ww = img_array.shape

        pred_mask = []
        for i in range(1, cc - 1):
            img_tmp = img_array[i - 1:i + 2]
            pred_slice = self.forward(img_tmp)
            pred_mask.append(pred_slice)
        pred_mask = np.array(pred_mask)

        if resize:
            pred_mask = zoom(pred_mask, [1, h/pred_mask.shape[1], w/pred_mask.shape[2]])

        return pred_mask.astype(np.uint8)

    def normalize(self, slice, bottom=99, down=1):
        b = np.percentile(slice, bottom)
        t = np.percentile(slice, down)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp

class ClsPredictor:

    def __init__(self, gpu, model_pth):
        self.gpu = gpu
        self.model_pth = model_pth
        self.num_instance_list = [12, -1, -1]
        self.data_shape = (1, 169, 169)
        self.load_model()

    def load_model(self):
        self.net = create_model('coat_lite_tiny_mil_multi_view_agg',
                                pretrained=True,
                                num_classes=2,
                                drop_rate=0.0,
                                drop_path_rate=0.0,
                                drop_block_rate=None,
                                in_chans=1,
                                img_size=169,
                                final_drop=0,
                                mode='eval')
        checkpoint = torch.load(self.model_pth, map_location='cpu')
        self.net.load_state_dict(checkpoint['model'])
        self.net.eval()
        self.net.cuda(self.gpu)

    def forward(self, img_array):
        with torch.no_grad(), torch.cuda.device(self.gpu):
            img_tensor = torch.from_numpy(img_array).cuda().float().unsqueeze(0)
            pred, _, _ = self.net(img_tensor)
            pred = F.softmax(pred, dim=1)
            pred = pred.cpu().numpy()
        return pred

    def predict(self, img_array, mask_array):
        sample = self.img_transform(img_array, mask_array)
        pred = self.forward(sample)
        return pred[0]

    def img_transform(self, img_array, mask_array):
        img_array = self.normalize(img_array)
        ori_c_, ori_h_, ori_w_ = img_array.shape
        non_zero = np.nonzero(mask_array)
        c_l, h_l, w_l = np.min(non_zero[0]), np.min(non_zero[1]), np.min(non_zero[2])
        c_bias, h_bias, w_bias = np.max(non_zero[0]) - c_l + 1, np.max(non_zero[1]) - h_l + 1, np.max(non_zero[2]) - w_l + 1
        w_r = np.minimum(int(w_l + w_bias), ori_w_)
        h_r = np.minimum(int(h_l + h_bias), ori_h_)
        c_r = np.minimum(int(c_l + c_bias), ori_c_)
        w_l = np.maximum(int(w_l), 0)
        h_l = np.maximum(int(h_l), 0)
        c_l = np.maximum(int(c_l), 0)
        c_diff = c_r - c_l
        if c_diff < self.num_instance_list[0]:
            c_l = np.maximum(int(np.floor((self.num_instance_list[0] - c_diff) / 2)), 0)
            c_r = np.maximum(int(np.ceil((self.num_instance_list[0] - c_diff) / 2)), ori_c_)

        img_ = img_array[c_l:c_r, h_l:h_r, w_l:w_r]

        axis_c_list, sections_c_list = slice_sample(img_.shape[0], self.num_instance_list[0])

        img_instance_c_list = []

        for axis_c in axis_c_list:
            img_patch_ = img_[axis_c, :, :]
            img_patch_ = get_sample(img_patch_, img_size=self.data_shape[2])
            img_instance_c_list.append(img_patch_)

        img_instance_list = img_instance_c_list
        num_instance = len(img_instance_list)
        sample = np.zeros((num_instance, self.data_shape[0], self.data_shape[1], self.data_shape[2]), dtype=np.float16)
        for i in range(num_instance):
            sample[i, :, :, :] = img_instance_list[i]

        return sample

    def normalize(self, slice, bottom=99, down=1):
        b = np.percentile(slice, bottom)
        t = np.percentile(slice, down)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp

def slice_sample(num_img_, num_instance):
    s_ids_ = list(np.linspace(0, num_img_, num_instance + 1))
    sections_list = []
    st_id_list = []
    for i in range(1, len(s_ids_)):
        start_ = int(s_ids_[i - 1])
        stop_ = int(s_ids_[i])
        if s_ids_[i - 1] - start_ > 0:
            start_ += 1
        if s_ids_[i] - stop_ > 0:
            stop_ += 1
        sections_list.append((start_, stop_))
        st_list = list(range(start_, stop_))
        np.random.shuffle(st_list)
        st_id_list.append(st_list[0])
    assert len(set(st_id_list)) == num_instance, 'error in slice sample'
    return st_id_list, sections_list

def get_sample(img_, img_size=64):
    img_ = center_pading_fix(img_, im_pad_val=0)
    if img_.shape[1] > img_size:
        img_ = crop_center(img_, img_size, img_size)
    else:
        img_ = pading_fix(img_, im_pad_val=0, pad_shape=img_size)
    return img_

def center_pading_fix(img, im_pad_val=0, pad_shape=None):
    img = img[np.newaxis,:,:]
    num_channel,height, width = img.shape
    if pad_shape==None:
        max_edge = np.max([height, width])
    else:
        max_edge = pad_shape

    random_bias_c_l = 0
    random_bias_c_r = 0

    random_bias_h_l = int((max_edge - height)//2 if (max_edge-height)>0 else 0)
    random_bias_h_r = int(max_edge-height-random_bias_h_l)

    random_bias_w_l = int((max_edge - width)//2 if (max_edge-width)>0 else 0)
    random_bias_w_r = int(max_edge-width-random_bias_w_l)

    new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r),
                                  (random_bias_h_l, random_bias_h_r),
                                  (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
    return new_img[0]

def crop_center(img, croph, cropw):
    height, width = img.shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[starth:starth+croph, startw:startw+cropw]

def pading_fix(img, im_pad_val=0, pad_shape=None):
    img = img[np.newaxis,:,:]

    num_channel,height, width = img.shape
    if pad_shape==None:
        max_edge = np.max([height, width])
    else:
        max_edge = pad_shape

    random_bias_c_l = 0
    random_bias_c_r = 0

    random_bias_h_l = 0
    random_bias_h_r = int(max_edge - height - random_bias_h_l)

    random_bias_w_l = 0
    random_bias_w_r = int(max_edge - width - random_bias_w_l)
    new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r),
                                  (random_bias_h_l, random_bias_h_r),
                                  (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)

    return new_img[0]