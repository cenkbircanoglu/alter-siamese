import random
import time
from collections import Counter

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

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

    tr_data_loader, te_data_loader = data_loaders()

    net = getattr(models, config.network).get_network()(config.network_channel)
    if config.cuda:
        net = net.cuda()

    net = train(net=net, loader=tr_data_loader)

    # evaluate(net, tr_data_loader)
    # evaluate(net, te_data_loader)

    torch.save(net, '%s/model.pt' % config.result_dir)

    create_embeddings(loader=tr_data_loader, net=net, outputfile='train')
    create_embeddings(loader=te_data_loader, net=net, outputfile='test')


def train(net, loader):
    criterion = getattr(losses, config.loss)()
    optimizer = optim.Adam(net.parameters())
    loss_history = []
    start = time.time()
    for epoch in range(0, config.epochs):
        epoch_loss = 0
        for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
            img, label = data
            if config.cuda:
                img, label = Variable(img).cuda(), Variable(label).cuda()
            else:
                img, label = Variable(img), Variable(label)
            output = net(img)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
        print('Epoch number: %s loss:%s' % (epoch, epoch_loss))
        loss_history.append(epoch_loss / loader.__len__())
    end = time.time()
    with open(config.log_path, "a") as f:
        f.write('%s %s\n' % (str(end - start), str(loss_history[-1])))

    show_plot(range(config.epochs), loss_history)
    return net


def evaluate(net, loader):
    counts = []
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        img, label = data
        if config.cuda:
            img, label = Variable(img).cuda(), Variable(label).cuda()
        else:
            img, label = Variable(img), Variable(label)
        output = net(img)
        # TODO
        for j, boolean in enumerate((output.data.numpy()), 0):
            counts.append(boolean[0] == bool(label.data[j][0]))

    counter = Counter(counts)
    with open(config.log_path, "a") as f:
        f.write('%s\n' % (str(counter)))


def create_embeddings(loader, net, outputfile):
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        img, label = data
        if config.cuda:
            img, label = Variable(img).cuda(), Variable(label).cuda()
        else:
            img, label = Variable(img), Variable(label)
        output = net(img)
        with open('%s/%s_embeddings.csv' % (config.result_dir, outputfile), 'a') as f:
            np.savetxt(f, output.data.numpy())
        with open('%s/%s_labels.csv' % (config.result_dir, outputfile), 'a') as f:
            np.savetxt(f, label.numpy())
