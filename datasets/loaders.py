import random

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from config import  get_config
from datasets.dataset import NetworkDataset
from datasets.siamese_dataset import SiameseNetworkDataset
from datasets.triplet import TripletDataset
from datasets.triplet_dataset import TripletNetworkDataset
from histogram import HistogramSampler

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

    val_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )

    val_data_loader = DataLoader(val_dataset,
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

    return tr_data_loader, val_data_loader, te_data_loader


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

    val_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        negative=config.negative,
        positive=config.positive,
        train=train
    )

    val_data_loader = DataLoader(val_dataset,
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
                                batch_size=config.batch_size)

    return tr_data_loader, val_data_loader, te_data_loader


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

    val_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )
    val_data_loader = DataLoader(val_triplet_dataset,
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
                                batch_size=config.batch_size)

    return tr_data_loader, val_data_loader, te_data_loader


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
    val_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train
    )

    sampler = HistogramSampler(val_dataset.labels, config.batch_size)
    val_data_loader = DataLoader(val_dataset,
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
    return tr_data_loader, val_data_loader, te_data_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--trainer', type=str, default="listwise")
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--data_name', type=str, default="mnist")
    parser.add_argument('--loader_name', type=str, default="data_loaders")
    parser.add_argument('--label_count', type=int, default=8)
    import torch

    torch.manual_seed(1137)
    np.random.seed(1137)
    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    trainer_name = kwargs['trainer']
    kwargs.pop('trainer')

    set_config(trainer_name, **kwargs)
    tr_data_loader, val_data_loader, te_data_loader = trip_loaders(trainer_name)

    for a in tr_data_loader:
        print(a)
