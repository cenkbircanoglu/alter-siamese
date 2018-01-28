import copy
import itertools
import random

import PIL.ImageOps
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets

from config import get_config, set_config

random.seed(1137)
np.random.seed(1137)

import random


def itershuffle(iterable, bufsize=10000):
    """Shuffle an iterator. This works by holding `bufsize` items back
    and yielding them sometime later. This is NOT 100% random, proved or anything."""
    iterable = iter(iterable)
    buf = []
    try:
        while True:
            for i in xrange(random.randint(1, bufsize - len(buf))):
                buf.append(iterable.next())
            random.shuffle(buf)
            for i in xrange(random.randint(1, bufsize)):
                if buf:
                    yield buf.pop()
                else:
                    break
    except StopIteration:
        random.shuffle(buf)
        while buf:
            yield buf.pop()
    raise StopIteration


class SiamesePairNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.create_pairs()

    def create_pairs(self):
        img1 = copy.deepcopy(self.image_folder_dataset.imgs)
        img2 = copy.deepcopy(self.image_folder_dataset.imgs)
        random.shuffle(img1)
        random.shuffle(img2)
        self.pairs = itershuffle(itertools.combinations(img1, 2))

    def __getitem__(self, index):
        img0_tuple, img1_tuple = next(self.pairs)
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
        return (len(self.image_folder_dataset.imgs) ** 2) / 2


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
    counter = 0
    for i in tr_data_loader:
        counter += 1
        print(counter, len(tr_data_loader))
