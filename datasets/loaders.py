import random

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from config import get_config
from datasets.dataset import NetworkDataset
from datasets.siamese_dataset import SiameseNetworkDataset
from datasets.triplet_dataset import TripletNetworkDataset
from histogram_dataset import HistogramSampler
random.seed(1137)
np.random.seed(1137)


def data_loaders(train=True):
    config = get_config()
    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )
    tr_data_loader = DataLoader(tr_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)

    te_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )
    te_data_loader = DataLoader(te_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)

    return tr_data_loader, te_data_loader


def pair_loaders(train=True):
    config = get_config()
    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_siamese_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        negative=config.negative,
        positive=config.positive,
        train=train
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
        positive=config.positive,
        train=train
    )
    te_data_loader = DataLoader(te_siamese_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=1)

    return tr_data_loader, te_data_loader


def triplet_loaders(train=True):
    config = get_config()
    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )
    tr_data_loader = DataLoader(tr_triplet_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=config.batch_size)

    te_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )
    te_data_loader = DataLoader(te_triplet_dataset,
                                shuffle=True,
                                num_workers=config.num_workers,
                                batch_size=1)

    return tr_data_loader, te_data_loader


def histogram_loaders(train=True):
    config = get_config()
    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )

    sampler = HistogramSampler(tr_dataset.labels, config.batch_size)

    tr_data_loader = DataLoader(tr_dataset,
                                batch_sampler=sampler,
                                num_workers=config.num_workers)
    te_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )

    sampler = HistogramSampler(te_dataset.labels, config.batch_size)
    te_data_loader = DataLoader(te_dataset,
                                batch_sampler=sampler,
                                num_workers=config.num_workers)
    return tr_data_loader, te_data_loader
