import random

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from config import get_config
from datasets.batch_sampler import BalancedBatchSampler
from datasets.dataset import NetworkDataset
from datasets.pair import SiamesePairNetworkDataset
from datasets.triplet_dataset import TripletNetworkDataset

random.seed(1137)
np.random.seed(1137)


def data_loaders(train=True, val=False):
    config = get_config()

    batch_size = config.batch_size
    if 'dense' in config.network:
        batch_size = 16

    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )
    tr_data_loader = DataLoader(tr_dataset,
                                shuffle=train,
                                num_workers=config.num_workers,
                                batch_size=batch_size)

    val_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )

    val_data_loader = DataLoader(val_dataset,
                                 shuffle=train,
                                 num_workers=config.num_workers,
                                 batch_size=batch_size)

    te_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )
    te_data_loader = DataLoader(te_dataset,
                                shuffle=train,
                                num_workers=config.num_workers,
                                batch_size=batch_size)

    return tr_data_loader, val_data_loader, te_data_loader


def pair_loaders(train=True, val=False):
    config = get_config()

    batch_size = config.batch_size
    if 'dense' in config.network:
        batch_size = 12

    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_siamese_dataset = SiamesePairNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        negative=config.negative,
        positive=config.positive,
        train=train,
        val=val
    )
    tr_data_loader = DataLoader(tr_siamese_dataset,
                                shuffle=train,
                                num_workers=config.num_workers,
                                batch_size=batch_size)

    val_dataset = SiamesePairNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        negative=config.negative,
        positive=config.positive,
        train=train,
        val=val,
        vall=train
    )

    val_data_loader = DataLoader(val_dataset,
                                 shuffle=train,
                                 num_workers=config.num_workers,
                                 batch_size=batch_size)

    te_siamese_dataset = SiamesePairNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        negative=config.negative,
        positive=config.positive,
        train=train,
        val=val
    )
    te_data_loader = DataLoader(te_siamese_dataset,
                                shuffle=train,
                                num_workers=config.num_workers,
                                batch_size=batch_size)

    return tr_data_loader, val_data_loader, te_data_loader


def triplet_loaders(train=True, val=False):
    config = get_config()
    batch_size = config.batch_size
    if 'dense' in config.network:
        batch_size = 8

    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.tr_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )
    tr_data_loader = DataLoader(tr_triplet_dataset,
                                shuffle=train,
                                num_workers=config.num_workers,
                                batch_size=batch_size)

    val_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val,
        vall=train
    )
    val_data_loader = DataLoader(val_triplet_dataset,
                                 shuffle=train,
                                 num_workers=config.num_workers,
                                 batch_size=batch_size)

    te_triplet_dataset = TripletNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )
    te_data_loader = DataLoader(te_triplet_dataset,
                                shuffle=train,
                                num_workers=config.num_workers,
                                batch_size=batch_size)

    return tr_data_loader, val_data_loader, te_data_loader


def online_pair_loaders(train=True, val=False):
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
        train=train,
        val=val
    )

    val_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )

    te_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )

    train_batch_sampler = BalancedBatchSampler(tr_dataset.image_folder_dataset, n_classes=8, n_samples=8)
    val_batch_sampler = BalancedBatchSampler(val_dataset.image_folder_dataset, n_classes=8, n_samples=8)
    test_batch_sampler = BalancedBatchSampler(te_dataset.image_folder_dataset, n_classes=8, n_samples=8)

    kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}
    tr_data_loader = DataLoader(tr_dataset, batch_sampler=train_batch_sampler, **kwargs)
    val_data_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **kwargs)
    te_data_loader = DataLoader(te_dataset, batch_sampler=test_batch_sampler, **kwargs)

    return tr_data_loader, val_data_loader, te_data_loader


def online_triplet_loaders(train=True, val=False):
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
        train=train,
        val=val
    )

    val_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.val_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )

    te_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root=config.te_dir),
        transform=transform,
        should_invert=False,
        channel=config.channel,
        train=train,
        val=val
    )

    train_batch_sampler = BalancedBatchSampler(tr_dataset.image_folder_dataset, n_classes=8, n_samples=8)
    val_batch_sampler = BalancedBatchSampler(val_dataset.image_folder_dataset, n_classes=8, n_samples=8)
    test_batch_sampler = BalancedBatchSampler(te_dataset.image_folder_dataset, n_classes=8, n_samples=8)

    kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}
    tr_data_loader = DataLoader(tr_dataset, batch_sampler=train_batch_sampler, **kwargs)
    val_data_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **kwargs)
    te_data_loader = DataLoader(te_dataset, batch_sampler=test_batch_sampler, **kwargs)

    return tr_data_loader, val_data_loader, te_data_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--trainer', type=str, default="triplet")
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--data_name', type=str, default="mnist")
    parser.add_argument('--loader_name', type=str, default="triplet_loaders")
    parser.add_argument('--label_count', type=int, default=8)
    parser.add_argument('--network', type=str, default='net28')
    import torch

    torch.manual_seed(1137)
    np.random.seed(1137)
    from config import set_config

    args = parser.parse_args()

    kwargs = vars(args)
    trainer_name = kwargs['trainer']
    kwargs.pop('trainer')

    set_config(trainer_name, **kwargs)
    print(get_config().__dict__)
    tr_data_loader, val_data_loader, te_data_loader = online_pair_loaders(train=True, val=False)

    for b in tr_data_loader:
        print(b)
