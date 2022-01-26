# from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from PIL import Image
from paddle import to_tensor
from models.networks import get_pad
from models.networks import ConvWithActivation
from models.networks import DeConvWithActivation


class Residual(nn.Layer):

    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3,
            padding=1, stride=strides)
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=3,
            padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1,
                stride=strides)
        self.batch_norm2d = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        return F.relu(out)


class STRnet2(nn.Layer):

    def __init__(self, n_in_channel=3):
        super(STRnet2, self).__init__()
        self.conv1 = ConvWithActivation(3, 32, kernel_size=4, stride=2,
            padding=1)
        self.conva = ConvWithActivation(32, 32, kernel_size=3, stride=1,
            padding=1)
        self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2,
            padding=1)
        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 64)
        self.res3 = Residual(64, 128, same_shape=False)
        self.res4 = Residual(128, 128)
        self.res5 = Residual(128, 256, same_shape=False)
        self.res6 = Residual(256, 256)
        self.res7 = Residual(256, 512, same_shape=False)
        self.res8 = Residual(512, 512)
        self.conv2 = ConvWithActivation(512, 512, kernel_size=1)
        self.deconv1 = DeConvWithActivation(512, 256, kernel_size=3,
            padding=1, stride=2)
        self.deconv2 = DeConvWithActivation(256 * 2, 128, kernel_size=3,
            padding=1, stride=2)
        self.deconv3 = DeConvWithActivation(128 * 2, 64, kernel_size=3,
            padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3,
            padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(64, 3, kernel_size=3, padding=1,
            stride=2)
        self.lateral_connection1 = nn.Sequential(nn.Conv2D(256, 256,
            kernel_size=1, padding=0, stride=1), nn.Conv2D(256, 512,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(512, 512,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(512, 256,
            kernel_size=1, padding=0, stride=1))
        self.lateral_connection2 = nn.Sequential(nn.Conv2D(128, 128,
            kernel_size=1, padding=0, stride=1), nn.Conv2D(128, 256,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(256, 256,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(256, 128,
            kernel_size=1, padding=0, stride=1))
        self.lateral_connection3 = nn.Sequential(nn.Conv2D(64, 64,
            kernel_size=1, padding=0, stride=1), nn.Conv2D(64, 128,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(128, 128,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(128, 64,
            kernel_size=1, padding=0, stride=1))
        self.lateral_connection4 = nn.Sequential(nn.Conv2D(32, 32,
            kernel_size=1, padding=0, stride=1), nn.Conv2D(32, 64,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(64, 64,
            kernel_size=3, padding=1, stride=1), nn.Conv2D(64, 32,
            kernel_size=1, padding=0, stride=1))
        self.conv_o1 = nn.Conv2D(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2D(32, 3, kernel_size=1)
        self.mask_deconv_a = DeConvWithActivation(512, 256, kernel_size=3,
            padding=1, stride=2)
        self.mask_conv_a = ConvWithActivation(256, 128, kernel_size=3,
            padding=1, stride=1)
        self.mask_deconv_b = DeConvWithActivation(256, 128, kernel_size=3,
            padding=1, stride=2)
        self.mask_conv_b = ConvWithActivation(128, 64, kernel_size=3,
            padding=1, stride=1)
        self.mask_deconv_c = DeConvWithActivation(128, 64, kernel_size=3,
            padding=1, stride=2)
        self.mask_conv_c = ConvWithActivation(64, 32, kernel_size=3,
            padding=1, stride=1)
        self.mask_deconv_d = DeConvWithActivation(64, 32, kernel_size=3,
            padding=1, stride=2)
        self.mask_conv_d = nn.Conv2D(32, 3, kernel_size=1)
        n_in_channel = 3
        cnum = 32
        self.coarse_conva = ConvWithActivation(n_in_channel, cnum,
            kernel_size=5, stride=1, padding=2)
        self.coarse_convb = ConvWithActivation(cnum, 2 * cnum, kernel_size=\
            4, stride=2, padding=1)
        self.coarse_convc = ConvWithActivation(2 * cnum, 2 * cnum,
            kernel_size=3, stride=1, padding=1)
        self.coarse_convd = ConvWithActivation(2 * cnum, 4 * cnum,
            kernel_size=4, stride=2, padding=1)
        self.coarse_conve = ConvWithActivation(4 * cnum, 4 * cnum,
            kernel_size=3, stride=1, padding=1)
        self.coarse_convf = ConvWithActivation(4 * cnum, 4 * cnum,
            kernel_size=3, stride=1, padding=1)
        self.astrous_net = nn.Sequential(ConvWithActivation(4 * cnum, 4 *
            cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4,
            padding=get_pad(64, 3, 1, 4)), ConvWithActivation(4 * cnum, 4 *
            cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16,
            padding=get_pad(64, 3, 1, 16)))
        self.coarse_convk = ConvWithActivation(4 * cnum, 4 * cnum,
            kernel_size=3, stride=1, padding=1)
        self.coarse_convl = ConvWithActivation(4 * cnum, 4 * cnum,
            kernel_size=3, stride=1, padding=1)
        self.coarse_deconva = DeConvWithActivation(4 * cnum * 3, 2 * cnum,
            kernel_size=3, padding=1, stride=2)
        self.coarse_convm = ConvWithActivation(2 * cnum, 2 * cnum,
            kernel_size=3, stride=1, padding=1)
        self.coarse_deconvb = DeConvWithActivation(2 * cnum * 3, cnum,
            kernel_size=3, padding=1, stride=2)
        self.coarse_convn = nn.Sequential(ConvWithActivation(cnum, cnum // 
            2, kernel_size=3, stride=1, padding=1), ConvWithActivation(cnum //
            2, 3, kernel_size=3, stride=1, padding=1, activation=None))
        self.c1 = nn.Conv2D(32, 64, kernel_size=1)
        self.c2 = nn.Conv2D(64, 128, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conva(x)
        con_x1 = x
        x = self.convb(x)
        x = self.res1(x)
        con_x2 = x
        x = self.res2(x)
        x = self.res3(x)
        con_x3 = x
        x = self.res4(x)
        x = self.res5(x)
        con_x4 = x
        x = self.res6(x)
        x_mask = x
        x = self.res7(x)
        x = self.res8(x)
        x = self.conv2(x)
        x = self.deconv1(x)
        x = paddle.concat([self.lateral_connection1(con_x4), x], axis=1)
        x = self.deconv2(x)
        x = paddle.concat([self.lateral_connection2(con_x3), x], axis=1)
        x = self.deconv3(x)
        xo1 = x
        x = paddle.concat([self.lateral_connection3(con_x2), x], axis=1)
        x = self.deconv4(x)
        xo2 = x
        x = paddle.concat([self.lateral_connection4(con_x1), x], axis=1)
        x = self.deconv5(x)
        x_o1 = self.conv_o1(xo1)
        x_o2 = self.conv_o2(xo2)
        x_o_unet = x
        mm = self.mask_deconv_a(paddle.concat([x_mask, con_x4], axis=1))
        mm = self.mask_conv_a(mm)
        mm = self.mask_deconv_b(paddle.concat([mm, con_x3], axis=1))
        mm = self.mask_conv_b(mm)
        mm = self.mask_deconv_c(paddle.concat([mm, con_x2], axis=1))
        mm = self.mask_conv_c(mm)
        mm = self.mask_deconv_d(paddle.concat([mm, con_x1], axis=1))
        mm = self.mask_conv_d(mm)
        mm = self.sig(mm)
        x = self.coarse_conva(x_o_unet)
        x = self.coarse_convb(x)
        x = self.coarse_convc(x)
        x_c1 = x
        x = self.coarse_convd(x)
        x = self.coarse_conve(x)
        x = self.coarse_convf(x)
        x_c2 = x
        x = self.astrous_net(x)
        x = self.coarse_convk(x)
        x = self.coarse_convl(x)
        x = self.coarse_deconva(paddle.concat([x, x_c2, self.c2(con_x2)], axis=1))
        x = self.coarse_convm(x)
        x = self.coarse_deconvb(paddle.concat([x, x_c1, self.c1(con_x1)], axis=1))
        x = self.coarse_convn(x)
        return x_o1, x_o2, x_o_unet, x, mm

if __name__ == '__main__':
    net = STRnet2()
    x = paddle.rand([1, 3, 64, 64])
    x_o1, x_o2, x_o_unet, x, mm = net(x)
    print(x.shape, mm.shape)
