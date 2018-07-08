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


class SiamesePairNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1, negative=0, positive=1,
                 train=True, val=False, vall=False):
        random.shuffle(image_folder_dataset.imgs)
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.negative = negative
        self.positive = positive
        self.num_inputs = 2
        self.num_targets = 1

        self.val = val
        self.labels = [x[1] for x in image_folder_dataset.imgs]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        self.vall = vall
        if self.vall:
            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.image_folder_dataset.imgs[i][1]]),
                               1]
                              for i in range(0, len(self.image_folder_dataset.imgs), 2)]

            negative_pairs = [[i,
                               np.random.choice(
                                   self.label_to_indices[np.random.choice(
                                       list(self.labels_set - set([self.image_folder_dataset.imgs[i][1]])))]),
                               0]
                              for i in range(0, len(self.image_folder_dataset.imgs), 2)]

            self.vall_pairs = positive_pairs + negative_pairs

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

    def get_train_items(self, index):
        if self.vall:
            img0 = self.image_folder_dataset.imgs[self.vall_pairs[index][0]][0]
            img1 = self.image_folder_dataset.imgs[self.vall_pairs[index][1]][0]
            target = self.vall_pairs[index][2]
        else:
            target = np.random.randint(0, 2)
            img0, label0 = self.image_folder_dataset.imgs[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label0])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label0])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img1 = self.image_folder_dataset.imgs[siamese_index][0]

        img0 = Image.open(img0)
        img1 = Image.open(img1)
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
        if target:
            label = torch.FloatTensor([float(self.positive)])
        else:
            label = torch.FloatTensor([float(self.negative)])
        return (img0, img1), label

    def __getitem__(self, index):
        if self.val:
            return self.get_val_items(index)
        return self.get_train_items(index)

    def __len__(self):
        if self.vall:
            return len(self.vall_pairs)
        return len(self.image_folder_dataset.imgs)


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
