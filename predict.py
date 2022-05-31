# 代码示例
# python predict.py /dataset/baidu/watermark_test_datasets/images  results

import os
import sys

import cv2
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import numpy as np
from test_dataloader import devdata
from models.sa_aidr import STRAIDR
import time
TIME = []

def pd_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    img = tensor.squeeze().cpu().numpy()
    img = img.clip(min_max[0], min_max[1])
    img = (img - min_max[0]) / (min_max[1] - min_max[0])
    if out_type == np.uint8:
        # scaling
        img = img * 255.0
    img = np.transpose(img, (1, 2, 0))
    img = img.round()
    img = img[:, :, ::-1]
    return img.astype(out_type)

def process(src_image_dir, save_dir):
    start = time.time()
    netG = STRAIDR(num_c=96)
    weights = paddle.load("model_best.pdparams")
    netG.load_dict(weights)

    netG.eval()

    Erase_data = devdata(dataRoot=src_image_dir, gtRoot=src_image_dir)
    Erase_data = DataLoader(Erase_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    print('OK!')

    for index, (imgs, path) in enumerate(Erase_data):
        _, _, h, w = imgs.shape
        if h < 1600 and w < 1600:  # 1000， 0.2399
            pad_size = 128
            h_padded = False
            w_padded = False
            if h % pad_size != 0:
                pad_h = pad_size - (h % pad_size)
                imgs = F.pad(imgs, (0, 0, 0, pad_h), mode='reflect')
                h_padded = True

            if w % pad_size != 0:
                pad_w = pad_size - (w % pad_size)
                imgs = F.pad(imgs, (0, pad_w, 0, 0), mode='reflect')
                w_padded = True
            print(index, imgs.shape, path)
            with paddle.no_grad():
                res = netG(imgs)[3]
                res += paddle.flip(netG(paddle.flip(imgs, axis=[2]))[3], axis=[2])
                res += paddle.flip(netG(paddle.flip(imgs, axis=[3]))[3], axis=[3])
                res += paddle.flip(netG(paddle.flip(imgs, axis=[2, 3]))[3], axis=[2, 3])
                res = res / 4  # 16 + 480
            if h_padded:
                res = res[:, :, 0:h, :]
            if w_padded:
                res = res[:, :, :, 0:w]
            res = pd_tensor2img(res)
            cv2.imwrite(os.path.join(save_dir, path[0]), res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            pad = 112
            m = nn.Pad2D(pad, mode='reflect')
            imgs = m(imgs)
            print(index, imgs.shape, path)
            _, _, h, w = imgs.shape
            step = 800
            res = paddle.zeros_like(imgs)
            for i in range(0, h, step):
                for j in range(0, w, step):
                    if h - i < step + 2 * pad:
                        i = h - (step + 2 * pad)
                    if w - j < step + 2 * pad:
                        j = w - (step + 2 * pad)
                    clip = imgs[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
                    clip = clip.cuda()
                    with paddle.no_grad():
                        g_images_clip = netG(clip)[3]
                        g_images_clip += paddle.flip(netG(paddle.flip(clip, axis=[2]))[3], axis=[2])
                        g_images_clip += paddle.flip(netG(paddle.flip(clip, axis=[3]))[3], axis=[3])
                        g_images_clip += paddle.flip(netG(paddle.flip(clip, axis=[2, 3]))[3], axis=[2, 3])
                        g_images_clip = g_images_clip / 4  # 16 + 480
                    res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = g_images_clip[:, :, pad:-pad, pad:-pad]
            res = res[:, :, pad:-pad, pad:-pad]
            res = pd_tensor2img(res)
            cv2.imwrite(os.path.join(save_dir, path[0]), res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('Total time: ', (time.time() - start) / len(Erase_data))

if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataRoot = src_image_dir
    savePath = save_dir

    # set gpu
    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu:0')
    else:
        paddle.set_device('cpu')
    process(src_image_dir, save_dir)
