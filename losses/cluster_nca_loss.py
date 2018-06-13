from __future__ import print_function, absolute_import

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.autograd import Variable


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist


class ClusterNCALoss(nn.Module):
    def __init__(self, alpha=16, n_cluster=16, beta=0.5):
        super(ClusterNCALoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_cluster
        self.use_cuda = False

    def cluster(self, inputs, targets):
        X = inputs.data.cpu().numpy()
        y = targets.data.cpu().numpy()

        y = [[t] for t in y]
        enc = OneHotEncoder()
        enc.fit(y)

        one_hot_y = enc.transform(y).toarray()
        X = np.concatenate([X, self.beta * one_hot_y], 1)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
        pred_cluster = kmeans.labels_
        result = dict()
        for i in range(len(y)):
            k = str(y[i]) + ' ' + str(pred_cluster[i])
            if k in result:
                result[k].append(i)
            else:
                result[k] = [i]
        split_ = result.values()
        return split_

    def forward(self, inputs, targets):
        split_ = self.cluster(inputs, targets)

        num_dim = inputs.size(1)
        n = inputs.size(0)
        centers = []
        inputs_list = []
        targets_ = []

        cluster_mat = np.ones([n, len(split_)])
        for i, split_i in enumerate(split_):
            size_ = len(split_i)
            if size_ > 1:
                for k in split_i:
                    cluster_mat[k][i] = float(size_ * size_) / ((size_ - 1) * (size_ - 1))
            targets_.append(targets[split_i[0]])
            input_ = torch.cat([inputs[i].resize(1, num_dim) for i in split_i], 0)
            centers.append(torch.mean(input_, 0))
            inputs_list.append(input_)

        if self.use_cuda:
            cluster_mat = Variable(torch.FloatTensor(cluster_mat)).cuda().detach()
        else:
            cluster_mat = Variable(torch.FloatTensor(cluster_mat)).detach()

        targets_ = torch.cat(targets_)

        centers = [center.resize(1, num_dim) for center in centers]
        centers = torch.cat(centers, 0)

        centers_dist = pair_euclidean_dist(inputs, centers) * cluster_mat
        loss = []
        num_match = 0
        for i, target in enumerate(targets):
            # for computation stability
            dist = centers_dist[i]
            pos_pair_mask = (targets_ == target)
            pos_pair = torch.masked_select(dist, pos_pair_mask)

            dist = torch.masked_select(dist, dist > 1e-3)
            pos_pair = torch.masked_select(pos_pair, pos_pair > 1e-3)

            base = (torch.max(dist) + torch.min(dist)).data[0] / 2
            pos_exp = torch.sum(torch.exp(-self.alpha * (pos_pair - base)))
            a_exp = torch.sum(torch.exp(-self.alpha * (dist - base)))
            loss_ = - torch.log(pos_exp / a_exp)
            loss.append(loss_)
            if loss_.data[0] < 0.3:
                num_match += 1

        return torch.mean(torch.cat(loss))

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))
