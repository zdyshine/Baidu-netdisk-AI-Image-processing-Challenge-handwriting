# from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
from models.non_local import NonLocalBlock
import paddle.nn.functional as F

class AIDR(nn.Layer):

    def __init__(self, in_channels=3, out_channels=3, num_c=48):
        super(AIDR, self).__init__()
        self.en_block1 = nn.Sequential(
            nn.Conv2D(in_channels, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2D(2),
            nn.Conv2D(num_c, num_c, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2D(num_c*2 + 256, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            NonLocalBlock(num_c*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2D(num_c*3 + 128, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2D(num_c*3 + 64, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2D(num_c*3, num_c*2, 3, padding=1,bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(num_c*2, num_c*2, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2,mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2D(num_c*2 + in_channels, 64, 3,padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(64, 32, 3, padding=1, bias_attr=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(32, out_channels, 3, padding=1, bias_attr=True))

    def forward(self, x, con_x2, con_x3, con_x4):
        # x -> x_o_unet: h, w
        # con_x1: h/2, w/2 # [1, 32, 32, 32]
        # con_x2: h/4, w/4 # [1, 64, 16, 16]
        # con_x3: h/8, w/8 # [1, 128, 8, 8]
        # con_x4: h/16, w/16 # [1, 256, 4, 4]
        pool1 = self.en_block1(x)      # h/2, w/2
        pool2 = self.en_block2(pool1)  # h/4, w/4
        pool3 = self.en_block3(pool2)  # h/8, w/8
        pool4 = self.en_block4(pool3)  # h/16, w/16
        # print('11111111111', con_x2.shape, con_x3.shape, con_x4.shape)
        # print('11111111111', pool2.shape, pool3.shape, pool4.shape)
        upsample5 = self.en_block5(pool4)
        concat5 = paddle.concat((upsample5, pool4, con_x4), axis=1)
        upsample4 = self.de_block1(concat5)
        concat4 = paddle.concat((upsample4, pool3, con_x3), axis=1)
        upsample3 = self.de_block2(concat4) # h/8, w/8
        concat3 = paddle.concat((upsample3, pool2, con_x2), axis=1)
        upsample2 = self.de_block3(concat3) # h/4, w/4
        concat2 = paddle.concat((upsample2, pool1), axis=1)
        upsample1 = self.de_block4(concat2) # h/2, w/2
        concat1 = paddle.concat((upsample1, x), axis=1)
        out = self.de_block5(concat1)
        return out

if __name__ == '__main__':
    bgr = paddle.rand([1, 3, 1920, 1280])
    bgr = paddle.to_tensor(bgr)
    model = AIDR(num_c=96)
    for _ in range(20):
        with paddle.no_grad():
            out = model(bgr)
        print(out.shape)

