import os
import cv2
import argparse
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import numpy as np
from data.dataloader import ErasingData, devdata
from models.sa_gan import STRnet2
from models.sa_aidr import STRAIDR

# paddle.enable_static()
parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='')
parser.add_argument('--pretrained', type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--savePath', type=str, default='./results/sn_tv/')
parser.add_argument('--net', type=str, default='str')
args = parser.parse_args()


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


# set gpu
if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu:0')
else:
    paddle.set_device('cpu')

batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
dataRoot = args.dataRoot
savePath = args.savePath
result_with_mask = savePath + 'WithMaskOutput/'
result_straight = savePath + 'StrOuput/'
# import pdb;pdb.set_trace()

if not os.path.exists(savePath):
    os.makedirs(savePath)
    os.makedirs(result_with_mask)
    os.makedirs(result_straight)

Erase_data = devdata(dataRoot=dataRoot, gtRoot=dataRoot)
Erase_data = DataLoader(Erase_data, batch_size=batchSize, shuffle=False, num_workers=args.numOfWorkers, drop_last=False)

# netG = STRAIDR(num_c=96)
if args.net == 'str':
    netG = STRnet2(3)
    weights = paddle.load('STE_best_38.6789.pdparams')
    netG.load_dict(weights)
    print('load:', 'STE_best_38.6789.pdparams')
    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False
elif args.net == 'idr':
    netG = STRAIDR(num_c=96)
    weights = paddle.load('STE_idr_38.0642.pdparams')
    netG.load_dict(weights)
    print('load:', 'STE_idr_38.0642.pdparams')
    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False
elif args.net == 'mix':
    netG1 = STRAIDR(num_c=96)
    netG2 = STRnet2(3)
    # weights1 = paddle.load('STE_best_37.99.pdparams') # 668
    weights1 = paddle.load('STE_idr_38.0642.pdparams') # 668
    # weights1 = paddle.load('STE_best.pdparams') # 668

    # weights2 = paddle.load('STE_best_38.6789.pdparams')
    weights2 = paddle.load('STE_best_38.6016_new.pdparams')
    netG1.load_dict(weights1)
    netG2.load_dict(weights2)
    print('load:', 'STE_idr_38.0642.pdparams', 'STE_best_38.6016_new.pdparams')
    netG1.eval()
    netG2.eval()
    for param in netG1.parameters():
        param.requires_grad = False
    for param in netG2.parameters():
        param.requires_grad = False

print('OK!')

import time
TIME = []

for index,(imgs, gt, path) in enumerate(Erase_data):
    pad = 106
    m = nn.Pad2D(pad, mode='reflect')
    imgs = m(imgs)
    print(index, imgs.shape, gt.shape, path)
    _, _, h, w = imgs.shape
    rh, rw = h, w
    step = 300
    res = paddle.zeros_like(imgs)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step + 2 * pad:
                i = h - (step + 2 * pad)
            if w - j < step + 2 * pad:
                j = w - (step + 2 * pad)
            clip = imgs[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
            clip = clip.cuda()
            start = time.time()
            with paddle.no_grad():
                if args.net == 'mix':
                    g_images_clip1 = netG1(clip)[3]
                    g_images_clip1 += paddle.flip(netG1(paddle.flip(clip, axis=[3]))[3], axis=[3])
                    g_images_clip1 = g_images_clip1 / 2
                    g_images_clip2 = netG2(clip)[3]
                    g_images_clip2 += paddle.flip(netG2(paddle.flip(clip, axis=[3]))[3], axis=[3])
                    g_images_clip2 = g_images_clip2 / 2
                    g_images_clip = (g_images_clip1 + g_images_clip2) / 2
                else:
                    g_images_clip = netG(clip)[3]
                    # g_images_clip += paddle.flip(netG(paddle.flip(clip, axis=[2]))[3], axis=[2])
                    g_images_clip += paddle.flip(netG(paddle.flip(clip, axis=[3]))[3], axis=[3])
                    # g_images_clip += paddle.flip(netG(paddle.flip(clip, axis=[2, 3]))[3], axis=[2, 3])
                    g_images_clip = g_images_clip / 2
            res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = g_images_clip[:, :, pad:-pad, pad:-pad]
    res = res[:, :, pad:-pad, pad:-pad]
    TIME.append(time.time() - start)
    # res = res.clamp_(0, 1)
    res = pd_tensor2img(res)
    cv2.imwrite(result_with_mask + path[0].replace('.jpg', '.png'), res)
print('total time: {}, avg_time: {}.'.format(np.sum(TIME), np.mean(TIME)))
