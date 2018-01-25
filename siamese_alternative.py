import time
from collections import Counter

import numpy as np

from dataset import MySiameseNetworkDataset
from network import MySiameseNetwork

np.random.seed(1137)

import random

random.seed(1137)
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import Config
from contrastive_loss import ContrastiveLoss
from draw_plot import show_plot, imshow

folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = MySiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                          transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                        transforms.ToTensor()
                                                                        ]), should_invert=False)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=Config.train_batch_size)

net = MySiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0
start = time.time()
for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img, label = data
        img, label = Variable(img), Variable(label)
        output1, output2 = net(img)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.data[0]))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])
end = time.time()
torch.save(net.state_dict(), "my_siamese_parameters")
print(end - start)

with open("siamese_alternative.txt", mode="a") as f:
    f.write("%s\n" % str(end - start))
show_plot(counter, loss_history, name="siamese_alternative")
folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = MySiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                          transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                        transforms.ToTensor()
                                                                        ])
                                          , should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)

train_counts = []
for i, data in enumerate(train_dataloader, 0):
    img, label = data
    img, label = Variable(img), Variable(label)
    output1, output2 = net(img)
    euclidean_distance = F.pairwise_distance(output1, output2)

    for j, boolean in enumerate((euclidean_distance.data < 2), 0):
        train_counts.append(boolean[0] == bool(label.data[j][0]))
train_counter = Counter(train_counts)
with open("siamese_alternative.txt", mode="a") as f:
    f.write("train %s\n" % str(train_counter))

test_counts = []
for i, data in enumerate(test_dataloader, 0):
    img, label = data
    img, label = Variable(img), Variable(label)
    output1, output2 = net(img)
    euclidean_distance = F.pairwise_distance(output1, output2)

    for j, boolean in enumerate((euclidean_distance.data < 2), 0):
        test_counts.append(boolean[0] == bool(label.data[j][0]))
test_counter = Counter(test_counts)
with open("siamese_alternative.txt", mode="a") as f:
    f.write("test %s\n" % str(test_counter))

dataiter = iter(test_dataloader)

for i in range(50):
    x1, label = next(dataiter)
    output1, output2 = net(Variable(x1))
    concatenated = torch.cat((x1[0][0:1].view(1, 1, 100, 100), x1[0][1:2].view(1, 1, 100, 100)), 0)
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),
           'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]), name="siamese_alternative")
