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


def GaussDistribution(data):
    """
    :param data:
    :return:
    """
    mean_value = torch.mean(data)
    diff = data - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return mean_value, std


class SoftmaxNeigLoss(nn.Module):
    def __init__(self, alpha=50,margin=0.1):
        super(SoftmaxNeigLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
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
        neg_dist = neg_dist.resize(len(neg_dist) // num_neg_instances, num_neg_instances)

        loss = list()

        for i, pos_pair in enumerate(pos_dist):
            pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_dist[i])[0]

            base = 1.0
            pos_logit = torch.sum(torch.exp(self.alpha * (base - pos_pair)))
            neg_logit = torch.sum(torch.exp(self.alpha * (base - neg_pair))) / 2

            loss_ = -torch.log(pos_logit / (pos_logit + neg_logit))
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

    print(SoftmaxNeigLoss(margin=0.1)(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
