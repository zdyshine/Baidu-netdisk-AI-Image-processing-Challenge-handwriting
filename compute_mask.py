import cv2
import os,glob, shutil
import numpy as np
path = './dataset/task2/dehw_train_dataset'
save_path = './dataset/task2/dehw_train_dataset/mask_331_25/'
gts = sorted(glob.glob(path + '/gts/*'))
images = sorted(glob.glob(path + '/images/*'))
os.makedirs(save_path, exist_ok=True)
# print(gts,images)
for gt_f,im_f in zip(gts,images):
    print(gt_f)
    gt = cv2.imread(gt_f)
    im = cv2.imread(im_f)
    # mask = np.where(abs(gt.astype(np.float32) - im.astype(np.float32)) > 40, 0, 1)
    kernel = np.ones((3,3),np.uint8) 
    # mask = cv2.erode(np.uint8(mask),  kernel, iterations=2)
    threshold = 25
    diff_image = np.abs(im.astype(np.float32) - gt.astype(np.float32))
    mean_image = np.mean(diff_image, axis=-1)
    mask = np.greater(mean_image, threshold).astype(np.uint8)
    mask = (1 - mask) * 255
    mask = cv2.erode(np.uint8(mask),  kernel, iterations=1)
    cv2.imwrite(save_path+os.path.basename(gt_f), np.uint8(mask))
    # print(gt[622,513],im[622,513],mask[622,513])
    # kernel = np.ones((2,2),np.uint8)  
    # erosion = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_OPEN, kernel)
    # # break
    # cv2.imshow('gt',gt) 
    # cv2.imshow('im',im) 
    # cv2.imshow('mask',np.uint8(mask)) 
    # # # cv2.imshow('erosion',np.uint8(erosion*255)) 
    # cv2.waitKey(0)
