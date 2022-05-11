import paddle
from paddle import nn
import paddle.nn.functional as F
# from tensorboardX import SummaryWriter

from PIL import Image
import numpy as np

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = paddle.bmm(feat, feat_t) / (ch * h * w)
    return gram

def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

def dice_loss(input, target):
    input = F.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    
    input = input 
    target = target

    a = paddle.sum(input * target, 1)
    b = paddle.sum(input * input, 1) + 0.001
    c = paddle.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = paddle.mean(d)
    return 1 - dice_loss

def bce_loss(input, target):
    input = F.sigmoid(input)

    input = input.reshape([input.shape[0], -1])
    target = target.reshape([target.shape[0], -1])
    
    input = input 
    target = target

    bce = paddle.nn.BCELoss()
    
    return bce(input, target)

class LossWithGAN_STE(nn.Layer):
    # def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
    def __init__(self, Lamda, lr, betasInit=(0.5, 0.9)):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()
        # self.extractor = extractor
        # self.discriminator = Discriminator_STE(3)    ## local_global sn patch gan
        # self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        # self.cudaAvailable = torch.cuda.is_available()
        # self.numOfGPUs = torch.cuda.device_count()
        # self.lamda = Lamda
        # self.writer = SummaryWriter(logPath)

    def forward(self, input, mask, x_o1,x_o2,x_o3,output,mm, gt, count, epoch):
        # self.discriminator.zero_grad()
        # D_real = self.discriminator(gt, mask)
        # D_real = D_real.mean().sum() * -1
        # D_fake = self.discriminator(output, mask)
        # D_fake = D_fake.mean().sum() * 1
        # D_loss = torch.mean(F.relu(1.+D_real)) + torch.mean(F.relu(1.+D_fake))  #SN-patch-GAN loss
        # D_fake = -torch.mean(D_fake)     #  SN-Patch-GAN loss

        # self.D_optimizer.zero_grad()
        # D_loss.backward(retain_graph=True)
        # self.D_optimizer.step()

        # self.writer.add_scalar('LossD/Discrinimator loss', D_loss.item(), count)
        
        # output_comp = mask * input + (1 - mask) * output
       # import pdb;pdb.set_trace()
        holeLoss =  self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = self.l1(mask * output, mask * gt)  
        mask_loss = bce_loss(mm, 1-mask)

        # GLoss = msrloss+ holeLoss + validAreaLoss+ prcLoss + styleLoss + 0.1 * D_fake + 1*mask_loss
        GLoss = mask_loss + holeLoss + validAreaLoss
        # self.writer.add_scalar('Generator/Joint loss', GLoss.item(), count)
        return GLoss.sum()
    
