import random

import PIL.ImageOps
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

random.seed(1137)
np.random.seed(1137)


class SiameseNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1, negative=0, positive=1):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.counter = 0,
        self.negative = negative
        self.positive = positive

    def __getitem__(self, index):
        img0_tuple = random.choice(self.image_folder_dataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if not should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.image_folder_dataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.image_folder_dataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        if self.channel == 1:
            img0 = img0.convert("L")
            img1 = img1.convert("L")
        elif self.channel == 3:
            img0 = img0.convert("RGB")
            img1 = img1.convert("RGB")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        if img1_tuple[1] != img0_tuple[1]:
            label = self.negative
        else:
            label = self.positive
        return (img0, img1), torch.from_numpy(np.array([int(label)], dtype=np.float32))

    def __len__(self):
        return len(self.image_folder_dataset.imgs)
