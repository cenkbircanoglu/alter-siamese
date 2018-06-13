from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

from losses.others.dist_weight_neighbour_loss import DistWeightNeighbourLoss


class BDWNeighbourLoss(nn.Module):
    def __init__(self, margin=0.7, slice=[0, 170, 341, 512]):
        super(BDWNeighbourLoss, self).__init__()
        self.s = slice
        self.margin = margin
        self.use_cuda = False

    def forward(self, inputs, targets):
        inputs = [inputs[:, self.s[i]:self.s[i + 1]] for i in range(len(self.s) - 1)]
        loss_list = []
        dist_weight_neighbour = DistWeightNeighbourLoss(margin=self.margin)
        if self.use_cuda:
            dist_weight_neighbour.cuda()
        for input in inputs:
            loss = dist_weight_neighbour(input, targets)
            loss_list.append(loss)

        return torch.mean(torch.cat(loss_list))

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
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(BDWNeighbourLoss(margin=1)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
