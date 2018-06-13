from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class NeighbourLoss(nn.Module):
    # It is actually the online version LMNN
    def __init__(self, k=1, margin=0.1):
        super(NeighbourLoss, self).__init__()
        self.k = k
        self.margin = margin
        self.use_cuda = False

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist_mat = euclidean_dist(inputs)
        eyes_ = Variable(torch.eye(n, n))
        if self.use_cuda:
            targets = targets.cuda()
            eyes_ = eyes_.cuda()
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_dist = torch.masked_select(dist_mat, pos_mask)
        neg_dist = torch.masked_select(dist_mat, neg_mask)

        num_instances = len(pos_dist) // n + 1
        num_neg_instances = n - num_instances

        pos_dist = pos_dist.resize(len(pos_dist) // (num_instances - 1), num_instances - 1)
        neg_dist = neg_dist.resize(
            len(neg_dist) // num_neg_instances, num_neg_instances)

        loss = list()

        for i, pos_pair in enumerate(pos_dist):

            pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_dist[i])[0]
            pos_pair = pos_pair[:self.k]

            neg_pair = torch.masked_select(neg_pair, neg_pair < pos_pair[-1] + self.margin)

            if len(neg_pair) > 0:
                loss.append(torch.mean(pos_pair) - torch.mean(neg_pair) + self.margin)
            else:
                continue

        if len(loss) == 0:
            loss = 0.0 * (torch.mean(pos_pair))
        else:
            loss = torch.sum(torch.cat(loss)) / n

        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))
