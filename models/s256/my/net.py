import torch.nn as nn


class MyNet(nn.Module):

    def __init__(self, channel, embedding_size=1000, **kwargs):
        super(MyNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=64, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=32, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 196, kernel_size=16, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(196),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(196, 256, kernel_size=8, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 392, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(392),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(392, 512, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, embedding_size),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self.forward_once(x)


def get_network():
    return MyNet


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    N = 4
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
