import torch.nn as nn


class MyNet(nn.Module):

    def __init__(self, channel, embedding_size=1000, **kwargs):
        super(MyNet, self).__init__()

        self.features1 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, 16, kernel_size=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=64),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=32),
            nn.ReLU(inplace=True),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=16),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=8),
            nn.ReLU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=4),
            nn.ReLU(inplace=True),
        )
        self.features6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, embedding_size),
        )

    def forward_once(self, x):
        x = self.features1(x)
        print(x.size())
        x = self.features2(x)
        print(x.size())
        x = self.features3(x)
        print(x.size())
        x = self.features4(x)
        print(x.size())
        x = self.features5(x)
        print(x.size())
        x = self.features6(x)
        print(x.size())
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self.forward_once(x)


def get_network():
    return MyNet


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    N = 2
    input_dim = 256
    output_dim = 10
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
