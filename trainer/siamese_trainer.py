import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

import losses
import models
from config import get_config
from datasets.loaders import pair_loaders, data_loaders
from utils.draw_plot import imshow, show_plot
from utils.make_dirs import create_dirs

random.seed(1137)
np.random.seed(1137)

config = get_config()


def run():
    create_dirs()
    with open(config.log_path, "a") as f:
        f.write('%s' % (str(config.__dict__)))

    tr_pair_loader, te_pair_loader = pair_loaders()

    net = getattr(models, config.network).get_network()(config.network_channel)
    if config.cuda:
        net = net.cuda()

    net = train(net=net, loader=tr_pair_loader)

    evaluate(net, tr_pair_loader)
    evaluate(net, te_pair_loader)

    visualize_distances(net, te_pair_loader)
    torch.save(net, '%s/model.pt' % config.result_dir)

    tr_data_loader, te_data_loader = data_loaders()

    create_embeddings(loader=tr_data_loader, net=net, outputfile='train')
    create_embeddings(loader=te_data_loader, net=net, outputfile='test')


def visualize_distances(net, loader):
    dataiter = iter(loader)
    (x0, _), _ = next(dataiter)

    for i in range(25):
        (_, x1), label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)
        if config.cuda:
            output = net((Variable(x0).cuda(), Variable(x1).cuda()))
        else:
            output = net((Variable(x0), Variable(x1)))
        euclidean_distance = F.pairwise_distance(output[0], output[1])
        imshow(torchvision.utils.make_grid(concatenated),
               'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]),
               "%s_%s" % (i, label2[0][0]))


def train(net, loader):
    criterion = getattr(losses, config.loss)()
    optimizer = optim.Adam(net.parameters())
    loss_history = []
    start = time.time()
    for epoch in range(0, config.epochs):
        epoch_loss = 0
        for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
            (img1, img2), label = data
            if config.cuda:
                img, label = (Variable(img1).cuda(), Variable(img2).cuda()), Variable(label).cuda()
            else:
                img, label = (Variable(img1), Variable(img2)), Variable(label)
            output = net(img)
            optimizer.zero_grad()
            loss_contrastive = criterion(output, label)
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


def evaluate(net, loader):
    counts = []
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        (img1, img2), label = data
        if config.cuda:
            img, label = (Variable(img1).cuda(), Variable(img2).cuda()), Variable(label).cuda()
        else:
            img, label = (Variable(img1), Variable(img2)), Variable(label)
        output = net(img)
        euclidean_distance = F.pairwise_distance(output[0], output[1])

        for j, boolean in enumerate((euclidean_distance.data >= config.margin), 0):
            counts.append(boolean[0] == bool(label.data[j][0]))

    counter = Counter(counts)
    with open(config.log_path, "a") as f:
        f.write('%s\n' % (str(counter)))


def create_embeddings(loader, net, outputfile):
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        img1, label = data
        if config.cuda:
            img = (Variable(img1).cuda(), Variable(img1).cuda())
        else:
            img = (Variable(img1), Variable(img1))
        output = net(img)
        with open('%s/%s_embeddings.csv' % (config.result_dir, outputfile), 'a') as f:
            np.savetxt(f, output[0].data.numpy())
        with open('%s/%s_labels.csv' % (config.result_dir, outputfile), 'a') as f:
            np.savetxt(f, label.numpy())
