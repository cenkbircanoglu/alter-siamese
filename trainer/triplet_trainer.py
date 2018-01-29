import random
import time

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

import losses
import models
from config import get_config
from datasets.loaders import data_loaders
from utils.draw_plot import show_plot
from utils.make_dirs import create_dirs

random.seed(1137)
np.random.seed(1137)

config = get_config()


def run():
    create_dirs()
    with open(config.log_path, "a") as f:
        f.write('%s' % (str(config.__dict__)))

    tr_triplet_loader, te_triplet_loader = triplet_loaders()

    net = getattr(models, config.network).get_network()(config.network_channel)
    if config.cuda:
        net = net.cuda()

    net = train(net=net, loader=tr_triplet_loader)

    torch.save(net, '%s/model.pt' % config.result_dir)

    tr_data_loader, te_data_loader = data_loaders()

    create_embeddings(loader=tr_data_loader, net=net, outputfile='train')
    create_embeddings(loader=te_data_loader, net=net, outputfile='test')


def train(net, loader):
    criterion = getattr(losses, config.loss)()
    optimizer = optim.Adam(net.parameters())
    loss_history = []
    start = time.time()
    for epoch in range(0, config.epochs):
        epoch_loss = 0
        for i, data in enumerate(loader, 0):
            anchor, pos, neg = data
            if config.cuda:
                img = (Variable(anchor).cuda(), Variable(pos).cuda(), Variable(neg).cuda())
            else:
                img = (Variable(anchor), Variable(pos), Variable(neg))
            output = net(img)
            optimizer.zero_grad()
            loss_contrastive = criterion(output)
            loss_contrastive.backward()
            optimizer.step()
            epoch_loss += loss_contrastive.data[0]
        print('Epoch number: %s loss:%s' % (epoch, epoch_loss))
        loss_history.append(epoch_loss)
    end = time.time()
    with open(config.log_path, "a") as f:
        f.write('%s %s\n' % (str(end - start), str(loss_history[-1])))

    show_plot(range(config.epochs), loss_history)
    return net


def create_embeddings(loader, net, outputfile):
    for i, data in enumerate(loader, 0):
        img1, label = data
        if config.cuda:
            img = (Variable(img1).cuda(), Variable(img1).cuda(), Variable(img1).cuda())
        else:
            img = (Variable(img1), Variable(img1), Variable(img1))
        output = net(img)
        with open('%s/%s_embeddings.csv' % (config.result_dir, outputfile), 'a') as f:
            np.savetxt(f, output[0].data.numpy())
        with open('%s/%s_labels.csv' % (config.result_dir, outputfile), 'a') as f:
            np.savetxt(f, label.numpy())
