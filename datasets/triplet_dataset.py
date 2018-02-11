import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

random.seed(1137)
np.random.seed(1137)


class TripletNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1, train=True):
        self.train = train
        if self.train:
            random.shuffle(image_folder_dataset.imgs)
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.data = [x[0] for x in image_folder_dataset.imgs]
        labels = [x[1] for x in image_folder_dataset.imgs]
        self.triplets = self.generate_triplets(labels, len(image_folder_dataset.imgs))
        self.num_inputs = 3
        self.num_targets = 0
        self.counter = 0

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels)
        n_classes = unique_labels.shape[0]

        for x in range(num_triplets):
            c1 = np.random.randint(0, n_classes - 1)
            c2 = np.random.randint(0, n_classes - 1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if len(indices[c1]) > 1 and len(indices[c2]) > 1:
                if len(indices[c1]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices[c1]) - 1)
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices[c1]) - 1)

                n3 = np.random.randint(0, len(indices[c2]) - 1)

                triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def get_eval_items(self, index):
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
        return (img0, img0, img0), img0_tuple[1]

    def get_train_items(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img)
            return img

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]
        a = Image.open(a)
        p = Image.open(p)
        n = Image.open(n)
        if self.channel == 1:
            a = a.convert("L")
            p = p.convert("L")
            n = n.convert("L")
        elif self.channel == 3:
            a = a.convert("RGB")
            p = p.convert("RGB")
            n = n.convert("RGB")

        # transform images if required
        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = transform_img(n)
        return img_a, img_p, img_n

    def __getitem__(self, index):
        if self.train:
            return self.get_train_items(index)
        return self.get_eval_items(index)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        return len(self.image_folder_dataset.imgs)
