import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, channel=1, embedding_size=128, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channel, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, embedding_size)

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x):
        x = self.forward_once(x)
        return F.log_softmax(x)
