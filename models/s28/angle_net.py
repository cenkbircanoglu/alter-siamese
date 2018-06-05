import torch.nn as nn
import torch.nn.functional as F

from losses.angle import AngleLinear


class Net(nn.Module):
    def __init__(self, channel=1, embedding_size=10, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channel, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = AngleLinear(50, embedding_size)

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self.forward_once(x)


def get_network():
    return Net


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    N = 4
    input_dim = 28
    output_dim = 10
    channel = 3
    model = get_network()(channel=channel, embedding_size=output_dim)

    x = Variable(torch.randn(N, channel, input_dim, input_dim))
    y = Variable(torch.randn(N, output_dim), requires_grad=False)

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(5):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        print(t, loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
