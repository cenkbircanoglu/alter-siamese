from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from tqdm import tqdm

from trainer import load_model_epoch


def run():
    from config import get_config
    config = get_config()
    load_model_epoch.run()
    return True
    print("Timer")
    import models
    from utils.make_dirs import create_dirs
    from datasets import loaders
    from torchsample.modules import ModuleTrainer
    create_dirs()
    cuda_device = -1
    tr_data_loader, val_data_loader, te_data_loader = loaders.online_pair_loaders()

    model = getattr(models, config.network).get_network()(channel=config.network_channel,
                                                          embedding_size=config.embedding)
    from losses.online_cosine import OnlineCosineLoss
    from datasets.data_utils import AllPositivePairSelector, HardNegativePairSelector

    margin = 0.5

    if args.selector == 'AllPositivePairSelector':
        criterion = OnlineCosineLoss(margin, AllPositivePairSelector())
    elif args.selector == 'HardNegativePairSelector':
        criterion = OnlineCosineLoss(margin, HardNegativePairSelector())

    if config.cuda:
        model.cuda()
        criterion.cuda()
    trainer = ModuleTrainer(model)
    trainer.compile(loss=criterion, optimizer='adam')
    if config.cuda:
        cuda_device = 0
    trainer.evaluate_loader(tr_data_loader, verbose=2, cuda_device=cuda_device)
    trainer.evaluate_loader(val_data_loader, verbose=2, cuda_device=cuda_device)
    trainer.evaluate_loader(te_data_loader, verbose=2, cuda_device=cuda_device)

    start_time = time.time()
    trainer.fit_loader(tr_data_loader, val_loader=val_data_loader, num_epoch=1, verbose=2,
                       cuda_device=cuda_device)
    end_time = time.time()
    with open("./times.log", mode="a") as f:
        f.write("%s %s\n" % (config.result_dir, str(end_time - start_time)))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--data_name', type=str, default="mnist")
    parser.add_argument('--network', type=str, default="net_28")
    parser.add_argument('--embedding', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--loss', type=str, default="OnlineCosineLossHardNegativePairSelector")
    parser.add_argument('--selector', type=str, default="HardNegativePairSelector")
    parser.add_argument('--epochs', type=int, default=10)

    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    set_config("siamese", **kwargs)

    run()
