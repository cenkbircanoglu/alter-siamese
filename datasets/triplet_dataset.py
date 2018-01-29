import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from configs.siam_att import SiamAtt


class TripletNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True, channel=1):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.channel = channel
        self.data = [x[0] for x in image_folder_dataset.imgs]
        labels = [x[1] for x in image_folder_dataset.imgs]
        self.triplets = self.generate_triplets(labels, len(image_folder_dataset.imgs))

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

    def __getitem__(self, index):
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

    def __len__(self):

        return 10 #self.triplets.size(0)


if __name__ == '__main__':
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader

    config = SiamAtt()
    tr_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transforms.Compose([
            transforms.Scale((config.heigth, config.width)),
            transforms.ToTensor()
        ]),
        should_invert=False,
        channel=config.channel
    )

    tr_data_loader = DataLoader(tr_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.tr_batch_size)

    te_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transforms.Compose([
            transforms.Scale((config.heigth, config.width)),
            transforms.ToTensor()
        ]),
        should_invert=False,
        channel=config.channel
    )

    te_data_loader = DataLoader(te_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.tr_batch_size)

    for i in tr_data_loader:
        print(i)
