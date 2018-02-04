import random

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from config import get_config
from datasets.dataset import NetworkDataset
from datasets.siamese_dataset import SiameseNetworkDataset
from datasets.triplet_dataset import TripletNetworkDataset

random.seed(1137)
np.random.seed(1137)

config = get_config()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def data_loaders():
    tr_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel
    )

    # tr_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    #                                        download=True, transform=transform)
    tr_data_loader = DataLoader(tr_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)

    te_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel
    )
    # te_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                        download=True, transform=transform)
    te_data_loader = DataLoader(te_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)

    return tr_data_loader, te_data_loader


def pair_loaders():
    tr_siamese_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        negative=config.negative,
        positive=config.positive
    )

    tr_data_loader = DataLoader(tr_siamese_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)

    te_siamese_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        negative=config.negative,
        positive=config.positive
    )

    te_data_loader = DataLoader(te_siamese_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=1)

    return tr_data_loader, te_data_loader


def triplet_loaders():
    tr_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel
    )

    tr_data_loader = DataLoader(tr_triplet_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)

    te_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel
    )

    te_data_loader = DataLoader(te_triplet_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=1)

    return tr_data_loader, te_data_loader
