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

random.seed(1137)
np.random.seed(1137)


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


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class SiamesePairNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1, negative=0, positive=1,
                 train=True, val=False):
        random.shuffle(image_folder_dataset.imgs)
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.negative = negative
        self.positive = positive
        self.num_inputs = 2
        self.num_targets = 1
        self.create_pairs()
        self.val = val


    def get_val_items(self, index):

        img0_tuple = self.image_folder_dataset.imgs[index]
        # we need to make sure approx 50% of images are in the same class

        img0 = Image.open(img0_tuple[0])
        if self.channel == 1:
            img0 = img0.convert("L")
        elif self.channel == 3:
            img0 = img0.convert("RGB")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)

        if self.transform is not None:
            img0 = self.transform(img0)
        return (img0, img0), img0_tuple[1]

    def create_pairs(self):
        img1 = copy.deepcopy(self.image_folder_dataset.imgs)
        random.shuffle(img1)
        self.pairs = []
        for x in batch(img1, 8):
            self.pairs.extend(list(itertools.combinations(x, 2)))

    def get_train_items(self, index):
        img0_tuple, img1_tuple = self.pairs[index]
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
        if img1_tuple[1] == img0_tuple[1]:
            label = torch.FloatTensor([float(self.positive)])
        else:
            label = torch.FloatTensor([float(self.negative)])
        return (img0, img1), label

    def __getitem__(self, index):
        if self.val:
            return self.get_val_items(index)
        return self.get_train_items(index)

    def __len__(self):
        if self.val:
            return len(self.image_folder_dataset.imgs)
        return len(self.pairs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--trainer', type=str, default="siamese")
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--data_name', type=str, default="mnist")
    parser.add_argument('--loader_name', type=str, default="data_loaders")
    parser.add_argument('--label_count', type=int, default=8)

    torch.manual_seed(1137)
    np.random.seed(1137)
    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    trainer_name = kwargs['trainer']
    kwargs.pop('trainer')

    set_config(trainer_name, **kwargs)

    config = get_config()
    tr_siamese_dataset = SiamesePairNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transforms.Compose([
            transforms.Scale((config.height, config.width)),
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
