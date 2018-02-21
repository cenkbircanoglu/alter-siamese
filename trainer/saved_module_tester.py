from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
from tqdm import tqdm


def run():
    from config import get_config
    config = get_config()
    print('%s/best_train_embeddings.csv' % config.result_dir)
    if os.path.exists('%s/best_train_embeddings.csv' % config.result_dir):
        return True
    print("Not Return")
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

    check_point = os.path.join(config.result_dir, "ckpt.pth.tar")
    if os.path.isfile(check_point):
        print("=> loading checkpoint '{}'".format(check_point))
        checkpoint = torch.load(check_point)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(check_point, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(check_point))
    criterion = getattr(losses, config.loss)()
    if config.cuda:
        model.cuda()
        criterion.cuda()
    trainer = ModuleTrainer(model)
    trainer.compile(loss=criterion, optimizer='adam')

    if config.cuda:
        cuda_device = 0

    tr_loss = trainer.evaluate_loader(tr_data_loader, cuda_device=cuda_device)
    val_loss = trainer.evaluate_loader(val_data_loader, cuda_device=cuda_device)
    te_loss = trainer.evaluate_loader(te_data_loader, cuda_device=cuda_device)
    with open(config.log_path, "a") as f:
        f.write('Best Train %s\nBest Val:%s\nBest Test:%s\n' % (str(tr_loss), str(val_loss), te_loss))

    tr_data_loader, val_data_loader, te_data_loader = getattr(loaders, config.loader_name)(train=False)
    tr_y_pred = trainer.predict_loader(tr_data_loader, cuda_device=cuda_device)
    save_embeddings(tr_y_pred, '%s/best_train_embeddings.csv' % config.result_dir)
    save_labels(tr_data_loader, '%s/best_train_labels.csv' % config.result_dir)

    val_y_pred = trainer.predict_loader(val_data_loader, cuda_device=cuda_device)
    save_embeddings(val_y_pred, '%s/best_val_embeddings.csv' % config.result_dir)
    save_labels(val_data_loader, '%s/best_val_labels.csv' % config.result_dir)

    te_y_pred = trainer.predict_loader(te_data_loader, cuda_device=cuda_device)
    save_embeddings(te_y_pred, '%s/best_test_embeddings.csv' % config.result_dir)
    save_labels(te_data_loader, '%s/best_test_labels.csv' % config.result_dir)


def save_embeddings(data, outputfile):
    with open(outputfile, 'a') as f:
        if type(data) == list:
            data = data[0]
        np.savetxt(f, data.data.cpu().numpy())


def save_labels(loader, outputfile):
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        img, label = data
        with open(outputfile, 'a') as f:
            np.savetxt(f, label.numpy())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--data_name', type=str, default="mnist")
    parser.add_argument('--network', type=str, default="net_28")
    parser.add_argument('--embedding', type=int, default=10)
    parser.add_argument('--loss', type=str, default="NLLLoss")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--negative', type=int, default=0)
    parser.add_argument('--positive', type=int, default=1)

    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    set_config("siamese", **kwargs)

    run()
