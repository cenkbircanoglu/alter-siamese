from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torchsample.metrics import CategoricalAccuracy
from tqdm import tqdm

from config import set_config


def run():
    from config import get_config
    config = get_config()
    print('%s/train_embeddings.csv' % config.result_dir)
    result_dir = config.result_dir#.replace("results", "best_results")
    print('%s/train_embeddings.csv' % result_dir)
    if os.path.exists('%s/train_embeddings.csv' % result_dir) and os.path.exists('%s/test_embeddings.csv' % result_dir):
        return True
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print("Saved Module Trainer Not Return")
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
    if os.path.isfile(check_point):
        print("=> loading checkpoint '{}'".format(check_point))
        checkpoint = torch.load(check_point)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(check_point, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(check_point))
    #criterion = getattr(losses, config.loss)()
    criterion = CrossEntropyLoss()
    if config.cuda:
        model.cuda()
        criterion.cuda()
    trainer = ModuleTrainer(model)

    trainer.compile(loss=criterion, optimizer='adam')

    if config.cuda:
        cuda_device = 0
    tr_data_loader, val_data_loader, te_data_loader = getattr(loaders, config.loader_name)(train=False, val=True)

    tr_y_pred = trainer.predict_loader(tr_data_loader, cuda_device=cuda_device)
    save_embeddings(tr_y_pred, '%s/train_embeddings.csv' % result_dir)
    save_labels(tr_data_loader, '%s/train_labels.csv' % result_dir)

    val_y_pred = trainer.predict_loader(val_data_loader, cuda_device=cuda_device)
    save_embeddings(val_y_pred, '%s/val_embeddings.csv' % result_dir)
    save_labels(val_data_loader, '%s/val_labels.csv' % result_dir)

    te_y_pred = trainer.predict_loader(te_data_loader, cuda_device=cuda_device)
    save_embeddings(te_y_pred, '%s/test_embeddings.csv' % result_dir)
    save_labels(te_data_loader, '%s/test_labels.csv' % result_dir)


def save_embeddings(data, outputfile):
    if not os.path.exists(os.path.dirname(outputfile)):
        os.makedirs(os.path.dirname(outputfile))
    with open(outputfile, 'a') as f:
        if type(data) == list:
            data = data[0]
        np.savetxt(f, data.data.cpu().numpy())


def save_labels(loader, outputfile):
    if not os.path.exists(os.path.dirname(outputfile)):
        os.makedirs(os.path.dirname(outputfile))
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        img, label = data
        with open(outputfile, 'a') as f:
            np.savetxt(f, label.numpy())
