import paddle
import numpy as np
import cv2
from os import listdir, walk
from os.path import join
import random
from PIL import Image

from paddle.vision.transforms import Compose, RandomCrop, ToTensor, CenterCrop
from paddle.vision.transforms import functional as F


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] =Image.fromarray(img_rotation)
    return imgs

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def ImageTransform():
    return Compose([
        # CenterCrop(size=loadSize),
        ToTensor(),
    ])
def ImageTransformTest(loadSize):
    return Compose([
        CenterCrop(size=loadSize),
        ToTensor(),
    ])

class PairedRandomCrop(RandomCrop):
    def __init__(self, size, keys=None):
        super().__init__(size, keys=keys)

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def _get_params(self, inputs):
        image = inputs[self.keys.index('image')]
        params = {}
        params['crop_prams'] = self._get_param(image, self.size)
        return params

    def _apply_image(self, img):
        i, j, h, w = self.params['crop_prams']
        return F.crop(img, i, j, h, w)

class ErasingData(paddle.io.Dataset):
    def __init__(self, dataRoot, loadSize, training=True, mask_dir='mask'):
        super(ErasingData, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform()
        self.training = training
        self.mask_dir = mask_dir
        self.RandomCropparam = RandomCrop(self.loadSize)

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        # print(self.imageFiles[index].replace('images', self.mask_dir).replace('jpg','png'))
        mask = Image.open(self.imageFiles[index].replace('images', self.mask_dir).replace('jpg','png'))
        gt = Image.open(self.imageFiles[index].replace('images','gts').replace('jpg','png'))
        # import pdb;pdb.set_trace()
        if self.training:
        # ### for data augmentation
            all_input = [img, mask, gt]
            all_input = random_horizontal_flip(all_input)   
            all_input = random_rotate(all_input)
            img = all_input[0]
            mask = all_input[1]
            gt = all_input[2]
        ### for data augmentation
        # param = RandomCrop.get_params(img.convert('RGB'), self.loadSize)
        param = self.RandomCropparam._get_param(img.convert('RGB'), self.loadSize)
        # print(param)
        inputImage = F.crop(img.convert('RGB'), *param)
        maskIn = F.crop(mask.convert('RGB'), *param)
        groundTruth = F.crop(gt.convert('RGB'), *param)
        del img
        del gt
        del mask

        inputImage = self.ImgTrans(inputImage)
        maskIn = self.ImgTrans(maskIn)
        groundTruth = self.ImgTrans(groundTruth)
        path = self.imageFiles[index].split('/')[-1]
       # import pdb;pdb.set_trace()

        return inputImage, groundTruth, maskIn, path
    
    def __len__(self):
        return len(self.imageFiles)

class devdata(paddle.io.Dataset):
    def __init__(self, dataRoot, gtRoot, loadSize=512):
        super(devdata, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.gtFiles = [join (gtRootK, files) for gtRootK, dn, filenames in walk(gtRoot) \
            for files in filenames if CheckImageFile(files)]
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform()
        # self.ImgTrans = ImageTransformTest(loadSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        gt = Image.open(self.gtFiles[index])
        # print(self.imageFiles[index],self.gtFiles[index])
        #import pdb;pdb.set_trace()
        inputImage = self.ImgTrans(img.convert('RGB'))

        groundTruth = self.ImgTrans(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]

        return inputImage, groundTruth,path
    
    def __len__(self):
        return len(self.imageFiles)