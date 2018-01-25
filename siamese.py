import time
from collections import Counter

import numpy as np

from network import SiameseNetwork

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
from dataset import SiameseNetworkDataset
from draw_plot import imshow, show_plot

folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=Config.train_batch_size)
margin = 1.0
net = SiameseNetwork()  # .cuda()
criterion = ContrastiveLoss(margin=margin)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

start = time.time()
for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = Variable(img0), Variable(img1), Variable(label)
        output1, output2 = net(img0, img1)
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
torch.save(net.state_dict(), "siamese_parameters")
print(end - start)

with open("siamese.txt", mode="a") as f:
    f.write("%s\n" % str(loss_contrastive.data[0]))
    f.write("%s\n" % str(end - start))

show_plot(counter, loss_history, name="siamese")
folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)

train_counts = []
data_counts = 0
for i, data in enumerate(train_dataloader, 0):
    img0, img1, label = data
    img0, img1, label = Variable(img0), Variable(img1), Variable(label)
    output1, output2 = net(img0, img1)
    euclidean_distance = F.pairwise_distance(output1, output2)

    for j, boolean in enumerate((euclidean_distance.data >= margin), 0):
        data_counts += 1
        train_counts.append(boolean[0] == bool(label.data[j][0]))
print(data_counts)
train_counter = Counter(train_counts)
with open("siamese.txt", mode="a") as f:
    f.write("train %s\n" % str(train_counter))

test_counts = []
for i, data in enumerate(test_dataloader, 0):
    img0, img1, label = data
    img0, img1, label = Variable(img0), Variable(img1), Variable(label)
    output1, output2 = net(img0, img1)
    euclidean_distance = F.pairwise_distance(output1, output2)

    for j, boolean in enumerate((euclidean_distance.data >= margin), 0):
        test_counts.append(boolean[0] == bool(label.data[j][0]))
test_counter = Counter(test_counts)
with open("siamese.txt", mode="a") as f:
    f.write("test %s\n" % str(test_counter))

dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(50):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0), Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),
           'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]), name="siamese")

net = SiameseNetwork()

train_counts = []
data_counts = 0
for i, data in enumerate(train_dataloader, 0):
    img0, img1, label = data
    img0, img1, label = Variable(img0), Variable(img1), Variable(label)
    output1, output2 = net(img0, img1)
    euclidean_distance = F.pairwise_distance(output1, output2)

    for j, boolean in enumerate((euclidean_distance.data >= margin), 0):
        data_counts += 1
        train_counts.append(boolean[0] == bool(label.data[j][0]))
print(data_counts)
train_counter = Counter(train_counts)
with open("siamese.txt", mode="a") as f:
    f.write("train %s\n" % str(train_counter))

test_counts = []
for i, data in enumerate(test_dataloader, 0):
    img0, img1, label = data
    img0, img1, label = Variable(img0), Variable(img1), Variable(label)
    output1, output2 = net(img0, img1)
    euclidean_distance = F.pairwise_distance(output1, output2)

    for j, boolean in enumerate((euclidean_distance.data >= margin), 0):
        test_counts.append(boolean[0] == bool(label.data[j][0]))
test_counter = Counter(test_counts)
with open("siamese.txt", mode="a") as f:
    f.write("test %s\n" % str(test_counter))
