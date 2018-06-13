from __future__ import absolute_import

import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from .KNNSoftmax import KNNSoftmax


def random_slice(nums):
    dim = sum(nums)
    index_ = list(range(dim))
    random.shuffle(index_)
    index_list = [index_[nums[i]:(nums[i] + nums[i + 1])]
                  for i in range(len(nums) - 1)]
    return index_list


class BranchKNNSoftmax(nn.Module):
    def __init__(self, alpha=40, k=100, nums=[0, 128, 128, 128, 128]):
        super(BranchKNNSoftmax, self).__init__()
        self.Nums = nums
        self.alpha = alpha
        self.K = k
        self.index_list = random_slice(self.Nums)

    def forward(self, inputs, targets):
        # index_list = random_slice(self.Nums)
        inputs = [inputs[:, k_index]
                  for k_index in self.index_list]
        loss_list, prec_list, pos_d_list, neg_d_list = [], [], [], []

        for input_ in inputs:
            norm = input_.norm(dim=1, p=2, keepdim=True)
            input_ = input_.div(norm.expand_as(input_))
            loss, prec_, pos_d, neg_d = KNNSoftmax(alpha=self.alpha)(input_, targets)
            loss_list.append(loss)
            prec_list.append(prec_)
            pos_d_list.append(pos_d)
            neg_d_list.append(neg_d)

        loss = torch.mean(torch.cat(loss_list))
        acc = np.mean(prec_list)
        pos_d = np.mean((pos_d_list))
        neg_d = np.mean((neg_d_list))

        return loss, acc, pos_d, neg_d
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
    # print(BranchKNNSoftmax()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
