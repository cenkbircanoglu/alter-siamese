import glob
import os

import cv2
import numpy

import numpy as np
import matplotlib.pyplot as plt

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list[:16]):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output


dir = "/media/cenk/2TB1/alter_siamese/data/aloi_red2_ill/**/**/"  # current directory
ext = ".png"  # whatever extension you want
import random

pathname = os.path.join(dir, "*" + ext)
imagelist = glob.glob(pathname)
random.shuffle(imagelist)
output = concat_n_images(imagelist)
cv2.imwrite("test.jpg", output)
