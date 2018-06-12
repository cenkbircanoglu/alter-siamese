import torch.nn as nn


class MyNet(nn.Module):

    def __init__(self, channel, embedding_size=256, **kwargs):
        super(MyNet, self).__init__()

        self.embedding = embedding_size
        self.features = nn.Sequential(
            nn.Conv2d(channel, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.embedding)
        return x

    def forward(self, x):
        return self.forward_once(x)


def get_network():
    return MyNet


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    N = 10
    input_dim = 32
    output_dim = 256
    channel = 3
    model = get_network()(channel=channel, embedding_size=output_dim)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(model, count_parameters(model))

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
