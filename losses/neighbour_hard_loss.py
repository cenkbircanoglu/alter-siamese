from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


# import numpy as np


class NeighbourHardLoss(nn.Module):
    def __init__(self, margin=0.05):
        super(NeighbourHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.use_cuda = False

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        eye_ = Variable(torch.eye(n))
        if self.use_cuda:
            eye_ = eye_.cuda()
        eye_ = eye_.eq(1)
        pos_mask = mask - eye_

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][pos_mask[i]].min())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        return self.ranking_loss(dist_an, dist_ap, y)

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    # print('training data is ', x)
    # print('initial parameters are ', w)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(NeighbourHardLoss(margin=0.1)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
