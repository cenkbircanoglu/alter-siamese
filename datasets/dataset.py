import random

import PIL.ImageOps
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

random.seed(1137)
np.random.seed(1137)


class NetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1):
        random.shuffle(image_folder_dataset.imgs)
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.counter = 0
        self.num_inputs = 1
        self.num_targets = 1

    def __getitem__(self, index):
        img0_tuple = self.image_folder_dataset.imgs[self.counter]
        # we need to make sure approx 50% of images are in the same class
        self.counter += 1
        img0 = Image.open(img0_tuple[0])
        if self.channel == 1:
            img0 = img0.convert("L")
        elif self.channel == 3:
            img0 = img0.convert("RGB")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)

        if self.transform is not None:
            img0 = self.transform(img0)
        return img0, img0_tuple[1]

    def __len__(self):
        return len(self.image_folder_dataset.imgs)
