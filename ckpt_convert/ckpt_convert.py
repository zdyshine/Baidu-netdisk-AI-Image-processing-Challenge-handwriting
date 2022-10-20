'''
训练的动态图模型转静态图模型
'''

import paddle
from paddle.static import InputSpec
from BiSeNetV2 import BiSeNetV2
from nafa_archv1 import NAFNet
########################################################################
model = BiSeNetV2(num_classes=2)
weights = paddle.load('../maskseg/output_enhance/best_model/model.pdparams') # 0923, 0.70293模型
model.load_dict(weights)
model.eval()

# step 2: 定义 InputSpec 信息
x_spec = InputSpec(shape=[1, 3, 1024, 1024], dtype='float32', name='x')

# step 3: 调用 jit.save 接口
net = paddle.jit.save(model, path='../submit/stac/seg', input_spec=[x_spec])  # 动静转换
########################################################################
model = NAFNet(img_channel=3, width=32, middle_blk_num=8,
               enc_blk_nums=[1, 1, 2, 2], dec_blk_nums=[2, 2, 1, 1], decmask_blk_nums=[1, 1, 1, 1])
weights = paddle.load("model_ema.pdparams")
model.load_dict(weights)
model.eval()

# step 2: 定义 InputSpec 信息
x_spec = InputSpec(shape=[1, 3, 480, 480], dtype='float32', name='x')

# step 3: 调用 jit.save 接口
net = paddle.jit.save(model, path='../submit/stac/ema', input_spec=[x_spec])  # 动静转换
# ########################################################################

# # 按照要求安装环境
# !pip install onnx==1.10.1 onnxruntime-gpu==1.10 paddle2onnx
#
# !paddle2onnx --model_dir ./stac_restormer --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 13 --save_file result_restormer.onnx