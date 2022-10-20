# -------------------------------------------------------------------------------------------------------------------
# 依赖导入
# -------------------------------------------------------------------------------------------------------------------

import os
import sys
import time
import math
import glob
import copy
import random

import cv2
import numpy as np
import pandas as pd

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from nafa_archv1 import NAFNet
Net = NAFNet(img_channel=3, width=32, middle_blk_num=8,
               enc_blk_nums=[1, 1, 2, 2], dec_blk_nums=[2, 2, 1, 1], decmask_blk_nums=[1, 1, 1, 1])

SEED = 42

paddle.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu:0')
else:
    paddle.set_device('cpu')

# -------------------------------------------------------------------------------------------------------------------
# EMA
# -------------------------------------------------------------------------------------------------------------------

def EMA(model, ema_model_path, model_path_list):

    ema_model = copy.deepcopy(model)
    ema_n = 0

    with paddle.no_grad():
        for _ckpt in model_path_list:
            model.load_dict(paddle.load(_ckpt))  # , map_location=torch.device('cpu')
            tmp_para_dict = dict(model.named_parameters())
            alpha = 1. / (ema_n + 1.)
            for name, para in ema_model.named_parameters():
                new_para = tmp_para_dict[name].clone() * alpha + para.clone() * (1. - alpha)
                para.set_value(new_para.clone())
            ema_n += 1

    paddle.save(ema_model.state_dict(), ema_model_path)
    print('ema finished !!!')

    return ema_model

# ---------------------------------------------------------------------------------------------------------------------------------
# 主函数定义
# ---------------------------------------------------------------------------------------------------------------------------------

def process():


    ema_model_path="./model_ema.pdparams"

    model_path_list = [f"../repaire/nafa_v1/99_0.7561.pdparams",  # best phase1
                       f"../repaire/nafa_v1/97_0.7560.pdparams",  # best phase1
                       f"../repaire/nafa_v1_psnr/50_0.7590.pdparams",  # best phase2
                       f"../repaire/nafa_v1_psnr/51_0.7588.pdparams",  # best phase1
    ]

    model = EMA(Net, ema_model_path, model_path_list)

    # paddle.jit.save(model, path='./stac/ema', input_spec=[paddle.static.InputSpec(shape=[1, 3, patch_size, patch_size], dtype='float32')])

# ---------------------------------------------------------------------------------------------------------------------------------
# 主函数调用
# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    process()


