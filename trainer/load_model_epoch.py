from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
from torchsample.metrics import CategoricalAccuracy
from tqdm import tqdm

from config import set_config


def run():
    from config import get_config
    config = get_config()
    print("Epochs")
    import losses
    import models
    from utils.make_dirs import create_dirs
    from datasets import loaders
    from torchsample.modules import ModuleTrainer
    create_dirs()
    cuda_device = -1

    model = getattr(models, config.network).get_network()(channel=config.network_channel,
                                                          embedding_size=config.embedding)

    check_point = os.path.join(config.result_dir, "ckpt.pth.tar")
    print("=> loading checkpoint '{}'".format(check_point))
    checkpoint = torch.load(check_point)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(check_point, checkpoint['epoch']))
    with open("./epochs.log", mode="a") as f:
        f.write("%s %s\n" % (config.result_dir, str(checkpoint['epoch'])))