import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from PIL import Image
from models.networks import get_pad, ConvWithActivation, DeConvWithActivation
from models.idr import AIDR

def img2photo(imgs):
    return ((imgs + 1) * 127.5).transpose(1, 2).transpose(2, 3).detach().cpu().numpy()


def visual(imgs):
    im = img2photo(imgs)
    Image.fromarray(im[0].astype(np.uint8)).show()


class Residual(nn.Layer):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = torch.nn.utils.spectral_norm(self.conv2)
        if not same_shape:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1,
                                   # self.conv3 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                   stride=strides)
            # self.conv3 = torch.nn.utils.spectral_norm(self.conv3)
        self.batch_norm2d = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        # out = out + x
        return F.relu(out)


class ASPP(nn.Layer):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2D((1, 1))
        self.conv = nn.Conv2D(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2D(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2D(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2D(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2D(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2D(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(paddle.concat([image_features, atrous_block1, atrous_block6,
                                                  atrous_block12, atrous_block18], axis=1))
        return net


class STRAIDR(nn.Layer):
    def __init__(self, n_in_channel=3, num_c=48):
        super(STRAIDR, self).__init__()
        #### U-Net ####
        # downsample
        self.conv1 = ConvWithActivation(3, 32, kernel_size=4, stride=2, padding=1)
        self.conva = ConvWithActivation(32, 32, kernel_size=3, stride=1, padding=1)
        self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2, padding=1)
        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 64)
        self.res3 = Residual(64, 128, same_shape=False)
        self.res4 = Residual(128, 128)
        self.res5 = Residual(128, 256, same_shape=False)
        # self.nn = ConvWithActivation(256, 512, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.res6 = Residual(256, 256)
        self.res7 = Residual(256, 512, same_shape=False)
        self.res8 = Residual(512, 512)
        self.conv2 = ConvWithActivation(512, 512, kernel_size=1)

        # upsample
        self.deconv1 = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2)
        self.deconv2 = DeConvWithActivation(256 * 2, 128, kernel_size=3, padding=1, stride=2)
        self.deconv3 = DeConvWithActivation(128 * 2, 64, kernel_size=3, padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3, padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(64, 3, kernel_size=3, padding=1, stride=2)

        # lateral connection
        self.lateral_connection1 = nn.Sequential(
            nn.Conv2D(256, 256, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(256, 512, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(512, 512, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(512, 256, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2D(128, 128, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(128, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(256, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(256, 128, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2D(64, 64, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(128, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(128, 64, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2D(32, 32, kernel_size=1, padding=0, stride=1),
            nn.Conv2D(32, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(64, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(64, 32, kernel_size=1, padding=0, stride=1), )

        # self.relu = nn.elu(alpha=1.0)
        self.conv_o1 = nn.Conv2D(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2D(32, 3, kernel_size=1)
        ##### U-Net #####

        ### ASPP ###
        # self.aspp = ASPP(512, 256)
        ### ASPP ###

        ### mask branch decoder ###
        self.mask_deconv_a = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2)
        self.mask_conv_a = ConvWithActivation(256, 128, kernel_size=3, padding=1, stride=1)
        self.mask_deconv_b = DeConvWithActivation(256, 128, kernel_size=3, padding=1, stride=2)
        self.mask_conv_b = ConvWithActivation(128, 64, kernel_size=3, padding=1, stride=1)
        self.mask_deconv_c = DeConvWithActivation(128, 64, kernel_size=3, padding=1, stride=2)
        self.mask_conv_c = ConvWithActivation(64, 32, kernel_size=3, padding=1, stride=1)
        self.mask_deconv_d = DeConvWithActivation(64, 32, kernel_size=3, padding=1, stride=2)
        self.mask_conv_d = nn.Conv2D(32, 3, kernel_size=1)
        ### mask branch ###

        ##### Refine sub-network ######
        self.refine = AIDR(num_c=num_c)
        self.c1 = nn.Conv2D(32, 64, kernel_size=1)
        self.c2 = nn.Conv2D(64, 128, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: 3, h, w
        # downsample
        x = self.conv1(x)  # 32, h/2,w/2
        x = self.conva(x)  # 32, h/2,w/2
        con_x1 = x
        # print('con_x1: ',con_x1.shape)
        # import pdb;pdb.set_trace()
        x = self.convb(x)  # 64, h/4,w/4
        x = self.res1(x)  # 64, h/4,w/4
        con_x2 = x
        # print('con_x2: ', con_x2.shape)
        x = self.res2(x)  # 64, h/4,w/4
        x = self.res3(x)  # 128, h/8,w/8
        con_x3 = x
        # print('con_x3: ', con_x3.shape)
        x = self.res4(x)  # 128, h/8,w/8
        x = self.res5(x)  # 256, h/16,w/16
        con_x4 = x
        # print('con_x4: ', con_x4.shape)
        x = self.res6(x)  # 256, h/16,w/16
        # x_mask = self.nn(con_x4)    ### for mask branch  aspp
        # x_mask = self.aspp(x_mask)     ###  for mask branch aspp
        x_mask = x  ### no aspp
        # print('x_mask: ', x_mask.shape)
        # import pdb;pdb.set_trace()
        x = self.res7(x)  # 512, h/32,w/32
        x = self.res8(x)  # 512, h/32,w/32
        x = self.conv2(x)  # 512, h/32,w/32
        # upsample
        x = self.deconv1(x)  # 256, h/16,w/16
        # print(x.shape,con_x4.shape, self.lateral_connection1(con_x4).shape)
        x = paddle.concat([self.lateral_connection1(con_x4), x], axis=1)  # 256 + 256
        x = self.deconv2(x)  # 512->128, h/8,w/8
        x = paddle.concat([self.lateral_connection2(con_x3), x], axis=1)  # 128 + 128
        x = self.deconv3(x)  # 256->64, h/4,w/4
        xo1 = x
        x = paddle.concat([self.lateral_connection3(con_x2), x], axis=1)  # 64 + 64
        x = self.deconv4(x)  # 128->32, h/2,w/2
        xo2 = x
        x = paddle.concat([self.lateral_connection4(con_x1), x], axis=1)  # 32 + 32
        # import pdb;pdb.set_trace()
        x = self.deconv5(x)  # 64->3, h, w
        x_o1 = self.conv_o1(xo1)  # 64->3, h/4,w/4
        x_o2 = self.conv_o2(xo2)  # 32->3, h/2,w/2
        x_o_unet = x

        ### mask branch ###
        mm = self.mask_deconv_a(paddle.concat([x_mask, con_x4], axis=1))  # 256 + 256 -> 256 , h/8,w/8
        mm = self.mask_conv_a(mm)  # 256 -> 128, h/8,w/8
        mm = self.mask_deconv_b(paddle.concat([mm, con_x3], axis=1))  # 128 + 128 -> 128, h/4,w/4
        mm = self.mask_conv_b(mm)  # 128 -> 64, h/4,w/4
        mm = self.mask_deconv_c(paddle.concat([mm, con_x2], axis=1))  # 64 + 64 -> 64, h/2, w/2
        mm = self.mask_conv_c(mm)  # 64 -> 32, h/2, w/2
        mm = self.mask_deconv_d(paddle.concat([mm, con_x1], axis=1))  # 32 +32 -> 32, h, w
        mm = self.mask_conv_d(mm)  # 32 -> 3, h, w
        mm = self.sig(mm)
        ### mask branch end ###

        ###refine sub-network
        x = self.refine(x_o_unet, con_x2, con_x3, con_x4)
        return x_o1, x_o2, x_o_unet, x, mm


if __name__ == '__main__':
    net = STRAIDR()
    x = paddle.rand([1, 3, 64, 64])
    x_o1, x_o2, x_o_unet, x, mm = net(x)
    print(x.shape, mm.shape)
