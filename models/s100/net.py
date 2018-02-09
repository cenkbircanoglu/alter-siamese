from torch import nn


class Net(nn.Module):
    def __init__(self, channel=1, embedding_size=128, **kwargs):
        super(Net, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, embedding_size)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, x):
        return self.forward_once(x)


def get_network():
    return Net

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    N = 4
    input_dim = 100
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
