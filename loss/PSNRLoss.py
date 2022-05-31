
import numpy as np
import paddle
import paddle.nn as nn

class PSNRLoss(nn.Layer):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)

    def forward(self, pred, target):
        assert len(pred.shape) == 4

        return self.loss_weight * self.scale * paddle.log(((pred - target) ** 2).mean(axis=(1, 2, 3)) + 1e-8).mean()

# import torch
# import torch.nn as nn
# 
# class PSNRLoss1(nn.Module):
# 
#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(PSNRLoss1, self).__init__()
#         assert reduction == 'mean'
#         self.loss_weight = loss_weight
#         self.scale = 10 / np.log(10)
# 
#     def forward(self, pred, target):
#         assert len(pred.size()) == 4
# 
#         return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


if __name__ == '__main__':
    import cv2
    import torch
    img1 = cv2.imread('../result/results/bg_image_00013_0001.jpg')
    img1 = cv2.resize(img1, (256, 256))
    img1 = img1[np.newaxis, :, :, :]
    
    img1_torch = torch.from_numpy(np.transpose(img1, (0, 3, 1, 2))) / 255.
    img1_paddle = paddle.to_tensor(np.transpose(img1, (0, 3, 1, 2))) / 255.

    
    img2 = cv2.imread('../result/results/bg_image_00016_0016.jpg')
    img2 = cv2.resize(img2, (256, 256))

    img2 = img2[np.newaxis, :, :, :]
    img2_torch = torch.from_numpy(np.transpose(img2, (0, 3, 1, 2))) / 255.
    img2_paddle = paddle.to_tensor(np.transpose(img2, (0, 3, 1, 2))) / 255.

    psnr1 = PSNRLoss1()
    print(psnr1(img2_torch, img1_torch))

    psnr = PSNRLoss()
    print(psnr(img2_paddle, img1_paddle))