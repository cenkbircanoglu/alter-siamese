from __future__ import absolute_import

import random

import torch
from torch import nn
from torch.autograd import Variable

from losses.dist_weight_neighbour_loss import DistWeightNeighbourLoss
# TODO

def random_slice(nums):
    dim = sum(nums)
    index_ = list(range(dim))
    random.shuffle(index_)
    index_list = [index_[nums[i]:(nums[i] + nums[i + 1])]
                  for i in range(len(nums) - 1)]
    return index_list


class EnsembleDWNeighbourLoss(nn.Module):
    def __init__(self, margin=1, nums=[0, 170, 171, 171]):
        super(EnsembleDWNeighbourLoss, self).__init__()
        self.Nums = nums
        self.margin = margin
        self.use_cuda = False

    def forward(self, inputs, targets):
        index_list = random_slice(self.Nums)
        inputs = [inputs[:, k_index] for k_index in index_list]
        loss_list, prec_list, pos_d_list, neg_d_list = [], [], [], []

        dist_weight_neighbour = DistWeightNeighbourLoss(margin=self.margin)
        if self.use_cuda:
            dist_weight_neighbour.cuda()
        for input_ in inputs:
            norm = input_.norm(dim=1, p=2, keepdim=True)
            input_ = input_.div(norm.expand_as(input_))
            loss = dist_weight_neighbour(input_, targets)
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

    print(EnsembleDWNeighbourLoss(margin=1)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
