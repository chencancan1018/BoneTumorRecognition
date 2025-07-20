import numpy as np

def pading_fix(img, mask, is_padding=True, mode='train', im_pad_val=0, pad_shape=None):
    # img = img[np.newaxis, :, :]
    # mask = mask[np.newaxis, :,:]
    if is_padding:
        num_channel, height, width = img.shape
        if pad_shape == None:
            max_edge = np.max([height, width])
        else:
            max_edge = pad_shape
        new_img = np.zeros((num_channel, max_edge, max_edge)) + im_pad_val
        new_mask = np.zeros((1, max_edge, max_edge), dtype=np.uint8)

        if mode == 'train':
            random_bias_c_l = 0
            random_bias_c_r = 0

            random_bias_h_l = int(np.random.randint(max_edge - height) if (max_edge - height) > 0 else 0)
            random_bias_h_r = int(max_edge - height - random_bias_h_l)

            random_bias_w_l = int(np.random.randint(max_edge - width) if (max_edge - width) > 0 else 0)
            random_bias_w_r = int(max_edge - width - random_bias_w_l)
            
            new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r), \
                                       (random_bias_h_l, random_bias_h_r),
                                       (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
            new_mask = np.lib.pad(mask, [(random_bias_c_l, random_bias_c_r),\
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=0)
        else:
            random_bias_c_l = 0
            random_bias_c_r = 0

            random_bias_h_l = 0
            random_bias_h_r = int(max_edge - height - random_bias_h_l)

            random_bias_w_l = 0
            random_bias_w_r = int(max_edge - width - random_bias_w_l)
            new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r), \
                                       (random_bias_h_l, random_bias_h_r),
                                       (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
            new_mask = np.lib.pad(mask, [(random_bias_c_l, random_bias_c_r),\
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=0)
        return new_img, new_mask
    else:
        return img, mask


def center_pading_fix(img, is_padding=True, im_pad_val=0, pad_shape=None):
    img = img[np.newaxis, :, :]
    # mask = mask[np.newaxis, :,:]
    if is_padding:
        num_channel, height, width = img.shape
        if pad_shape == None:
            max_edge = np.max([height, width])
        else:
            max_edge = pad_shape
        new_img = np.zeros((1, max_edge, max_edge)) + im_pad_val

        random_bias_c_l = 0
        random_bias_c_r = 0

        random_bias_h_l = int((max_edge - height) // 2 if (max_edge - height) > 0 else 0)
        random_bias_h_r = int(max_edge - height - random_bias_h_l)

        random_bias_w_l = int((max_edge - width) // 2 if (max_edge - width) > 0 else 0)
        random_bias_w_r = int(max_edge - width - random_bias_w_l)

        new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r), \
                                   (random_bias_h_l, random_bias_h_r),
                                   (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
        return new_img[0]
    else:
        return img[0]


def random_crop_to_fix_shape(img, w_l, h_l, c_l, w_bias, h_bias, c_bias, extend_num=4, fix_shape=(3, 64, 64)):
    c, h, w = img.shape
    cf, hf, wf = fix_shape
    c_center = c_l + c_bias // 2
    h_center = h_l + h_bias // 2
    w_center = w_l + w_bias // 2
    startc = c_center - (cf // 2)
    # startc = startc + int(np.round(np.random.normal(0, 80/3., 1)))
    startc = startc+np.random.randint(-4,4)
    startc = np.max([startc, 0])
    starth = h_center - (hf // 2)
    starth = starth + int(np.round(np.random.normal(0, 50., 1)))
    # starth = starth + np.random.randint(-80, 80)
    starth = np.max([starth, 0])
    startw = w_center - (wf // 2)
    startw = startw + int(np.round(np.random.normal(0, 50., 1)))
    # startw = startw + np.random.randint(-80, 80)
    startw = np.max([startw, 0])
    if startc + cf >= c:
        bias_ = startc + cf - c
        startc = startc - bias_
    if starth + hf >= h:
        bias_ = starth + hf - h
        starth = starth - bias_
    if startw + wf >= w:
        bias_ = startw + wf - w
        startw = startw - bias_
    box = (startc, starth, startw, startc + fix_shape[0], starth + fix_shape[1], startw + fix_shape[2])
    return box
    # c, h, w = img.shape
    #
    # cc = np.random.randint(0, c - fix_shape[0])
    # hh = np.random.randint(0, h - fix_shape[1])
    # ww = np.random.randint(0, w - fix_shape[2])
    #
    # box = (cc, hh, ww, cc + fix_shape[0], hh + fix_shape[1], ww + fix_shape[2])
    # return box

def crop_center_3d(img, w_l, h_l, c_l, w_bias, h_bias, c_bias, extend_num=4, fix_shape=(3, 64, 64)):
    c, h, w = img.shape
    cf, hf, wf = fix_shape
    c_center = c_l+c_bias//2
    h_center = h_l+h_bias//2
    w_center = w_l+w_bias//2
    startc = c_center - (cf // 2)
    startc = np.max([startc,0])
    starth = h_center - (hf // 2)
    starth = np.max([starth,0])
    startw = w_center - (wf // 2)
    startw = np.max([startw,0])
    if startc + cf >= c:
        bias_ = startc+cf-c
        startc = startc-bias_
    if starth + hf >= h:
        bias_ = starth+hf-h
        starth = starth-bias_
    if startw + wf >= w:
        bias_ = startw + wf - w
        startw = startw - bias_
    box = (startc, starth, startw, startc + fix_shape[0], starth + fix_shape[1], startw + fix_shape[2])
    return box

def pading_fix_3d(img, mask, is_padding=True, mode='train', im_pad_val=0, num_instance_list=[3,3,3]):
    if is_padding:
        num_channel,height, width = img.shape
        # max_edge = np.max([height, width, num_channel])
        new_shape = []
        for i in range(len(num_instance_list)):
            axi_num_instance_ = num_instance_list[i]
            cur_num_ = img.shape[i]
            if cur_num_<axi_num_instance_:
                new_shape.append(axi_num_instance_)
            else:
                new_shape.append(cur_num_)


        new_img = np.zeros(new_shape, dtype=np.float32) + im_pad_val
        new_mask = np.zeros(new_shape, dtype=np.uint8)

        if mode == 'train':
            random_bias_c_l = int(np.random.randint(new_shape[0] - num_channel) if (new_shape[0]-num_channel)>0 else 0)
            random_bias_c_r = int(new_shape[0]-num_channel-random_bias_c_l)

            random_bias_h_l = int(np.random.randint(new_shape[1] - height) if (new_shape[1]-height)>0 else 0)
            random_bias_h_r = int(new_shape[1]-height-random_bias_h_l)

            random_bias_w_l = int(np.random.randint(new_shape[2] - width) if (new_shape[2]-width)>0 else 0)
            random_bias_w_r = int(new_shape[2]-width-random_bias_w_l)

            new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r),\
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
            new_mask = np.lib.pad(mask, [(random_bias_c_l, random_bias_c_r),\
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=0)
        else:
            random_bias_c_l = 0
            random_bias_c_r = int(new_shape[0] - num_channel - random_bias_c_l)

            random_bias_h_l = 0
            random_bias_h_r = int(new_shape[1] - height - random_bias_h_l)

            random_bias_w_l = 0
            random_bias_w_r = int(new_shape[2] - width - random_bias_w_l)
            new_img = np.lib.pad(img, [(random_bias_c_l, random_bias_c_r), \
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=im_pad_val)
            new_mask = np.lib.pad(mask, [(random_bias_c_l, random_bias_c_r),\
                                          (random_bias_h_l, random_bias_h_r),
                                          (random_bias_w_l, random_bias_w_r)], 'constant', constant_values=0)
            assert new_mask.shape==new_mask.shape, 'error in padding 3d'
        return new_img, new_mask
    else:
        return img,mask
