import random
import random

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from config import get_config, set_config
from datasets.dataset import SiameseNetworkDataset

import PIL.ImageOps
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import itertools

random.seed(1137)
np.random.seed(1137)


class SiamesePairNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.counter = 0
        self.create_pairs()

    def create_pairs(self):
        self.pairs = list(itertools.product(self.image_folder_dataset.imgs, self.image_folder_dataset.imgs))

    def __getitem__(self, index):
        img0_tuple, img1_tuple = random.choice(self.pairs)

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

        return (img0, img1), torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    set_config("SiamAtt")
    config = get_config()
    tr_siamese_dataset = SiamesePairNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transforms.Compose([
            transforms.Scale((config.heigth, config.width)),
            transforms.ToTensor()
        ]),
        should_invert=False,
        channel=config.channel
    )

    tr_data_loader = DataLoader(tr_siamese_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.tr_batch_size)
    for i in tr_data_loader:
        print(i)
