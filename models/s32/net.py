import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, channel=3, embedding_size=128, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, embedding_size)

    def forward_once(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_once(x)


def get_network():
    return Net


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    N = 4
    input_dim = 32
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
