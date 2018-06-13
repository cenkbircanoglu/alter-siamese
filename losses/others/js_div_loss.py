from __future__ import absolute_import

import random

import torch
import torch.nn.functional as F
from torch import nn


def random_slice(nums):
    dim = sum(nums)
    index_ = list(range(dim))
    random.shuffle(index_)
    index_list = [index_[nums[i]:(nums[i] + nums[i + 1])]
                  for i in range(len(nums) - 1)]
    return index_list


class JSDivLoss(nn.Module):
    def __init__(self, alpha=40, beta=40, k=100, nums=[0, 128, 128, 128, 128]):
        super(JSDivLoss, self).__init__()
        self.Nums = nums
        self.alpha = alpha
        self.beta = beta
        self.K = k
        self.index_list = random_slice(self.Nums)

    def forward(self, inputs, targets):
        inputs = [inputs[:, k_index] for k_index in self.index_list]
        dist_list = []

        # compute dist
        for input_ in inputs:
            norm = input_.norm(dim=1, p=2, keepdim=True)
            input_ = input_.div(norm.expand_as(input_))
            dist_ = euclidean_dist(input_)
            dist_list.append(dist_)

        # compute JS divergence
        JSDiv = []
        num_branch = len(dist_list)
        idx_list = gen_idx(num_branch)
        for pair in idx_list:
            a = -self.beta * dist_list[pair[0]]
            b = -self.beta * dist_list[pair[1]]
            js_div = compute_js_div(a, b)
            JSDiv.append(js_div)
        return torch.mean(JSDiv)

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


def gen_idx(num_branch):
    idx = []
    for i in range(num_branch - 1):
        for j in range(i + 1, num_branch):
            idx.append((i, j))
    return idx


def compute_js_div(a, b):
    num = float(a.size()[0])
    criterion = nn.KLDivLoss(size_average=False)
    softmax_a = F.softmax(a)
    softmax_b = F.softmax(b)
    softmax_mean = (softmax_a + softmax_b) / 2

    lsm_a = F.log_softmax(a)
    lsm_b = F.log_softmax(b)

    div = (0.5 / num) * (criterion(lsm_a, softmax_mean) + criterion(lsm_b, softmax_mean))
    return div


def euclidean_dist(inputs_):
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    # for numerical stability
    dist = dist.clamp(min=1e-12).sqrt()
    return dist
