import os
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from data.dataloader import ErasingData,devdata
from loss.Loss import LossWithGAN_STE
from models.sa_gan import STRnet2
from models.sa_aidr import STRAIDR
import utils
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=16, help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='', help='path for saving models')
parser.add_argument('--logPath', type=str, default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512, help='image loading size')
parser.add_argument('--dataRoot', type=str, default='')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=5000, help='epochs')
parser.add_argument('--net', type=str, default='str')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--lr_decay_iters', type=int, default=400000, help='learning rate decay per N iters')
parser.add_argument('--mask_dir', type=str, default='mask')
parser.add_argument('--seed', type=int, default=2022)
args = parser.parse_args()

log_file = os.path.join('./log', args.net + '_log.txt')
logging = utils.setup_logger(output=log_file, name=args.net)
logging.info(args)

# set gpu
if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu:0')
else:
    paddle.set_device('cpu')

# set random seed
logging.info('========> Random Seed: {}'.format(args.seed))
random.seed(args.seed)
np.random.seed(args.seed)
paddle.seed(args.seed)
paddle.framework.random._manual_program_seed(args.seed)


batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)

if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

dataRoot = args.dataRoot

Erase_data = ErasingData(dataRoot, loadSize, training=True, mask_dir=args.mask_dir)
Erase_data = DataLoader(Erase_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False)
val_dataRoot='./dataset/task2/dehw_val_dataset/images'
Erase_val_data = devdata(dataRoot=val_dataRoot, gtRoot=val_dataRoot.replace('images','gts'))
Erase_val_data = DataLoader(Erase_val_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
print('==============', len(Erase_val_data))
print('==============>net use: ', args.net)
if args.net == 'str':
    netG = STRnet2(3)
elif args.net == 'idr':
    netG = STRAIDR(num_c=96)

if args.pretrained != '':
    print('loaded ')
    weights = paddle.load(args.pretrained)
    netG.load_dict(weights)

count = 1
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=args.lr_decay_iters, gamma=args.gamma, verbose=False)
G_optimizer = paddle.optimizer.Adam(scheduler, parameters=netG.parameters(), weight_decay=0.0)#betas=(0.5, 0.9))

criterion = LossWithGAN_STE(lr=0.00001, betasInit=(0.0, 0.9), Lamda=10.0)
print('OK!')
num_epochs = args.num_epochs
mse = nn.MSELoss()
best_psnr = 0
iters = 0
for epoch in range(1, num_epochs + 1):
    netG.train()

    for k, (imgs, gt, masks, path) in enumerate(Erase_data):
        iters += 1
        #print(imgs.max(), gt.max(), masks.max())

        x_o1, x_o2, x_o3, fake_images, mm = netG(imgs)
        G_loss = criterion(imgs, masks, x_o1, x_o2, x_o3, fake_images, mm, gt, count, epoch)
        G_loss = G_loss.sum()
        G_optimizer.clear_grad()
        G_loss.backward()
        G_optimizer.step()
        scheduler.step()
        if iters % 100 == 0:
            logging.info('[{}/{}] Generator Loss of epoch{} is {:.5f}, {}, {},  Lr:{}'.format(iters, len(Erase_data) * num_epochs, epoch, G_loss.item(), args.net, args.mask_dir, G_optimizer.get_lr()))
        count += 1
    
        if (iters % 5000 == 0):
            netG.eval()
            val_psnr = 0
            for index, (imgs, gt, path) in enumerate(Erase_val_data):
                print(index, imgs.shape,gt.shape, path)
                _,_,h,w = imgs.shape
                rh, rw = h, w
                step = 512
                pad_h = step - h if h < step else 0
                pad_w = step - w if w < step else 0
                m = nn.Pad2D((0, pad_w,0, pad_h))
                imgs = m(imgs)
                _, _, h, w = imgs.shape
                res = paddle.zeros_like(imgs)
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        if h - i < step:
                            i = h - step
                        if w -j < step:
                            j = w - step
                        clip = imgs[:, :, i:i+step, j:j+step]
                        clip = clip.cuda()
                        with paddle.no_grad():
                            _, _, _, g_images_clip,mm = netG(clip)
                        g_images_clip = g_images_clip.cpu()
                        mm = mm.cpu()
                        clip = clip.cpu()
                        mm = paddle.where(F.sigmoid(mm)>0.5, paddle.zeros_like(mm), paddle.ones_like(mm))
                        g_image_clip_with_mask = clip * (mm) + g_images_clip * (1- mm)
                        res[:, :, i:i+step, j:j+step] = g_image_clip_with_mask
                res = res[:, :, :rh, :rw]
                output = utils.pd_tensor2img(res)
                target = utils.pd_tensor2img(gt)
                del res
                del gt
                psnr = utils.compute_psnr(target, output)
                del target
                del output
                val_psnr += psnr
                logging.info('index:{} psnr: {}'.format(index, psnr))
            ave_psnr = val_psnr/(index+1)
            paddle.save(netG.state_dict(), args.modelsSavePath + '/STE_{}_{:.4f}.pdparams'.format(epoch, ave_psnr))
            if ave_psnr > best_psnr:
                best_psnr = ave_psnr
                paddle.save(netG.state_dict(), args.modelsSavePath + '/STE_best.pdparams')
            logging.info('epoch: {}, ave_psnr: {}, best_psnr: {}'.format(epoch, ave_psnr, best_psnr))
