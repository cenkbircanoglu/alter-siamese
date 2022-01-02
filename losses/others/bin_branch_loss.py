from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

from losses.others.bin_deviance_loss import BinDevianceLoss


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class BinBranchLoss(nn.Module):
    def __init__(self, margin=0.5, slice=[0, 170, 341, 512]):
        super(BinBranchLoss, self).__init__()
        self.s = slice
        self.margin = margin
        self.use_cuda = False

    def forward(self, inputs, targets):
        inputs = [inputs[:, self.s[i]:self.s[i + 1]] for i in range(len(self.s) - 1)]
        loss_list = []
        bin_deviance = BinDevianceLoss(margin=self.margin)
        if self.use_cuda:
            bin_deviance.cuda()

        for input in inputs:
            loss = bin_deviance(input, targets)
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
    input_dim = 8
    output_dim = 512
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(BinBranchLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')