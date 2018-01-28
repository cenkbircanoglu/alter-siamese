import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from torch import optim
from torch.autograd import Variable

import models
from config import get_config
from datasets.loaders import data_loaders
from losses.contrastive_loss import ContrastiveLoss
from utils.draw_plot import imshow, show_plot
from utils.make_dirs import create_dirs

random.seed(1137)
np.random.seed(1137)


def run():
    config = get_config()
    create_dirs()
    with open(config.log_path, "a") as f:
        f.write('%s' % (str(config.__dict__)))

    tr_data_loader, te_data_loader = data_loaders()

    net = getattr(models, config.network).get_network()(config.network_channel)
    if config.cuda:
        net = net.cuda()
    criterion = ContrastiveLoss(margin=config.margin)
    optimizer = optim.Adam(net.parameters())

    ######### TRAIN #########
    counter = []
    loss_history = []
    iteration_number = 0
    start = time.time()
    for epoch in range(0, config.epochs):
        for i, data in enumerate(tr_data_loader, 0):
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
            if i % 25 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.data[0]))
                iteration_number += 25
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
    end = time.time()
    print(end - start)
    with open(config.log_path, "a") as f:

        f.write('%s %s\n' % (str(end - start), str(loss_history[-1])))

    show_plot(counter, loss_history)
    ######### TRAIN #########
    ########### EVALUATE ###########
    train_counts = []
    for i, data in enumerate(tr_data_loader, 0):
        (img1, img2), label = data
        if config.cuda:
            img, label = (Variable(img1).cuda(), Variable(img2).cuda()), Variable(label).cuda()
        else:
            img, label = (Variable(img1), Variable(img2)), Variable(label)
        output = net(img)
        euclidean_distance = F.pairwise_distance(output[0], output[1])

        for j, boolean in enumerate((euclidean_distance.data >= config.margin), 0):
            train_counts.append(boolean[0] == bool(label.data[j][0]))

    train_counter = Counter(train_counts)

    test_counts = []
    for i, data in enumerate(te_data_loader, 0):
        (img1, img2), label = data
        if config.cuda:
            img, label = (Variable(img1).cuda(), Variable(img2).cuda()), Variable(label).cuda()
        else:
            img, label = (Variable(img1), Variable(img2)), Variable(label)
        output = net(img)
        euclidean_distance = F.pairwise_distance(output[0], output[1])

        for j, boolean in enumerate((euclidean_distance.data >= config.margin), 0):
            test_counts.append(boolean[0] == bool(label.data[j][0]))
    test_counter = Counter(test_counts)
    with open(config.log_path, "a") as f:

        f.write('%s %s\n' % (str(train_counter), str(test_counter)))
    ########### EVALUATE ###########
    ########### VISUALIZE ###########

    dataiter = iter(te_data_loader)
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
        ########### VISUALIZE ###########
    torch.save(net, '%s/model.pt' % config.result_dir)
