from __future__ import absolute_import

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from .KNNSoftmax import KNNSoftmax


def random_slice(nums):
    dim = sum(nums)
    index_ = list(range(dim))
    cum_nums = np.cumsum(nums)
    random.shuffle(index_)
    index_list = [index_[cum_nums[i]: cum_nums[i + 1]]
                  for i in range(len(nums) - 1)]
    return index_list


class JSDivKNNSoftmaxLoss(nn.Module):
    def __init__(self, alpha=40, beta=40, gama=1, sigma=10, k=100, nums=[0, 128, 128, 128, 128]):
        super(JSDivKNNSoftmaxLoss, self).__init__()
        self.Nums = nums
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.sigma = sigma
        self.K = k
        self.index_list = random_slice(self.Nums)
        self.use_cuda = False

    def forward(self, inputs, targets):
        inputs = [inputs[:, k_index] for k_index in self.index_list]
        dist_list = []
        knnsoftmax_loss_list, prec_list, pos_d_list, neg_d_list = [], [], [], []

        for input_ in inputs:
            norm = input_.norm(dim=1, p=2, keepdim=True)
            input_ = input_.div(norm.expand_as(input_))
            dist_ = euclidean_dist(input_)

            # delete the diagonal elements of dist mat
            n = dist_.size()[0]
            eyes_ = Variable(torch.eye(n, n))
            if self.use_cuda:
                eyes_ = eyes_.cuda()

            mask_ = eyes_ == 0
            dist_ = torch.masked_select(dist_, mask=mask_)
            dist_ = dist_.resize(n, n - 1)

            dist_list.append(dist_)

            loss, prec_, pos_d, neg_d = KNNSoftmax(alpha=self.alpha, k=self.K)(input_, targets)
            knnsoftmax_loss_list.append(loss)
            prec_list.append(prec_)
            pos_d_list.append(pos_d)
            neg_d_list.append(neg_d)

        knnsoftmax_loss = torch.mean(torch.cat(knnsoftmax_loss_list))

        JSDiv = []
        num_branch = len(dist_list)
        idx_list = gen_idx(num_branch)
        for pair in idx_list:
            a = -self.beta * dist_list[pair[0]]
            b = -self.beta * dist_list[pair[1]]
            js_div = compute_js_div(a, b)
            JSDiv.append(js_div)

        div_loss = torch.exp(-self.sigma * (torch.cat(JSDiv) - 1))
        div_loss = torch.mean(div_loss)
        loss = knnsoftmax_loss + self.gama * div_loss
        return loss

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


def KLDiv(log_inputs, target):
    return torch.sum(target * (torch.log(target) - log_inputs))


def compute_js_div(a, b):
    num = float(a.size()[0])
    softmax_a = F.softmax(a)
    softmax_b = F.softmax(b)
    softmax_mean = (softmax_a + softmax_b) / 2

    lsm_a = F.log_softmax(a)
    lsm_b = F.log_softmax(b)

    div = (0.5 / num) * (KLDiv(lsm_a, softmax_mean) + KLDiv(lsm_b, softmax_mean))
    return div


def euclidean_dist(inputs_):
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    # for numerical stability
    dist = dist.clamp(min=1e-12).sqrt()
    return dist
