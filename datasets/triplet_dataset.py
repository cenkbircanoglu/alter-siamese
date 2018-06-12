import random

import PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

random.seed(1137)
np.random.seed(1137)


class TripletNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1, train=True, val=False, vall=False):
        self.train = train
        if self.train:
            random.shuffle(image_folder_dataset.imgs)
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.data = [x[0] for x in image_folder_dataset.imgs]
        self.labels = [x[1] for x in image_folder_dataset.imgs]
        self.labels_set = set(self.labels)
        self.num_inputs = 3
        self.num_targets = 0
        self.val = val
        self.vall = vall
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}

        if self.vall:
            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.image_folder_dataset.imgs[i][1]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.image_folder_dataset.imgs[i][1]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.image_folder_dataset.imgs))]
            self.triplets = triplets

    def get_train_items(self, index):

        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img)
            return img
        if self.vall:
            a = self.image_folder_dataset.imgs[self.triplets[index][0]][0]
            p = self.image_folder_dataset.imgs[self.triplets[index][1]][0]
            n = self.image_folder_dataset.imgs[self.triplets[index][2]][0]
        else:
            a, label1 = self.image_folder_dataset.imgs[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            p, label2 = self.image_folder_dataset.imgs[positive_index]
            n, label3 = self.image_folder_dataset.imgs[negative_index]

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

    def get_val_items(self, index):
        img0_tuple = self.image_folder_dataset.imgs[index]
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

    def __getitem__(self, index):
        if self.val:
            return self.get_val_items(index)
        return self.get_train_items(index)

    def __len__(self):
        return len(self.image_folder_dataset.imgs)
