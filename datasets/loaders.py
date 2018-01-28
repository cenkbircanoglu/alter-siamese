import random

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from config import get_config
from datasets.dataset import  SiameseNetworkDataset

random.seed(1137)
np.random.seed(1137)

config = get_config()


def data_loaders():
    tr_siamese_dataset = SiameseNetworkDataset(
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

    te_siamese_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transforms.Compose([
            transforms.Scale((config.heigth, config.width)),
            transforms.ToTensor()
        ]),
        should_invert=False,
        channel=config.channel
    )

    te_data_loader = DataLoader(te_siamese_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.te_batch_size)
    return tr_data_loader, te_data_loader
