import numpy as np

from datasets.dataset import SiameseNetworkDataset

np.random.seed(1137)

import random

random.seed(1137)
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader

from config import Config

from draw_plot import imshow

import os

if not os.path.exists("pairs/siamese"):
    os.makedirs("pairs/siamese")
if not os.path.exists("pairs/my_siamese"):
    os.makedirs("pairs/my_siamese")
folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Scale((Config.height, Config.width)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False, channel=Config.channel,
                                        concat=False)

vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=8)

for i, example_batch in enumerate(vis_dataloader, 0):
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated), name="pairs/siamese", i=i)

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Scale((Config.height, Config.width)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False, channel=Config.channel,
                                        concat=True)

vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=8)

for i, example_batch in enumerate(vis_dataloader, 0):
    concatenated = torch.cat((example_batch[0][:, 0:1], example_batch[0][:, 1:2]), 0)
    imshow(torchvision.utils.make_grid(concatenated), name="pairs/my_siamese", i=i)
