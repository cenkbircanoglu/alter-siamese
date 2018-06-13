from __future__ import print_function, absolute_import

import torch
from torch import nn


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist


class MCALoss(nn.Module):
    def __init__(self, alpha=16, centers=None, cluster_counter=None, center_labels=None):
        super(MCALoss, self).__init__()
        self.alpha = alpha
        self.centers = centers
        self.center_labels = center_labels
        self.cluster_counter = cluster_counter

    def forward(self, inputs, targets, _mask):
        centers_dist = pair_euclidean_dist(inputs, self.centers)
        loss = []
        for i, target in enumerate(targets):
            # for computation stability
            dist = centers_dist[i]

            pos_pair_mask = (self.center_labels == target)
            # print(pos_pair_mask[:7])
            neg_pair_mask = (self.center_labels != target)

            pos_pair = torch.masked_select(dist, pos_pair_mask)

            # count the closest cluster
            pos_idx = torch.sort(pos_pair)[1][0].data[0]
            self.cluster_counter[target.data[0]][pos_idx] += 1

            # delete the dead cluster
            pos_pair = torch.masked_select(pos_pair, _mask[target.data[0]])
            neg_pair = torch.sort(torch.masked_select(dist, neg_pair_mask))

            # only consider neighbor negative clusters
            neg_pair = neg_pair[0][:32]

            base = (torch.max(neg_pair) + torch.min(dist)).data[0] / 2
            pos_exp = torch.sum(torch.exp(-self.alpha * (pos_pair - base)))
            neg_exp = torch.sum(torch.exp(-self.alpha * (neg_pair - base)))
            loss_ = - torch.log(pos_exp / (pos_exp + neg_exp))
            loss.append(loss_)

        return torch.mean(torch.cat(loss))

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))
