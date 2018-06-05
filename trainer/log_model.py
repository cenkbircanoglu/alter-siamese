from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from torch.nn import Parameter
from tqdm import tqdm

from torch.nn import Parameter
def run():
    from config import get_config
    config = get_config()

    import losses
    import models
    from utils.make_dirs import create_dirs
    from datasets import loaders
    from torchsample.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from torchsample.metrics import CategoricalAccuracy
    from torchsample.modules import ModuleTrainer
    create_dirs()

    model = getattr(models, config.network).get_network()(channel=config.network_channel,
                                                          embedding_size=config.embedding)
    criterion = getattr(losses, config.loss)()
    if config.cuda:
        model.cuda()
        criterion.cuda()
    trainer = ModuleTrainer(model)

    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(config.result_dir, save_best_only=True, verbose=1),
                 CSVLogger("%s/logger.csv" % config.result_dir)]
    metrics = []
    if config.loader_name == 'data_loaders':
        metrics.append(CategoricalAccuracy(top_k=1))
    trainer.compile(loss=criterion, optimizer='adam', metrics=metrics)
    trainer.set_callbacks(callbacks)

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    with open("models.log", mode="a") as f:
        f.write("%s\n" % str(config.__dict__))
        f.write("%s\n" % str(model))
        f.write("%s\n" % str(get_n_params(model)))
        f.write("\n")


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
