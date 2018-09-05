from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from tqdm import tqdm


def run():
    from config import get_config
    config = get_config()
    print("Timer")
    import losses
    import models
    from utils.make_dirs import create_dirs
    from datasets import loaders
    from torchsample.modules import ModuleTrainer
    create_dirs()
    cuda_device = -1
    tr_data_loader, val_data_loader, te_data_loader = getattr(loaders, config.loader_name)(train=True)

    model = getattr(models, config.network).get_network()(channel=config.network_channel,
                                                          embedding_size=config.embedding)
    criterion = getattr(losses, config.loss)()
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
    parser.add_argument('--loss', type=str, default="NLLLoss")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--negative', type=int, default=0)
    parser.add_argument('--positive', type=int, default=1)

    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    set_config("siamese", **kwargs)

    run()
