# coding=utf-8
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


def similarity(inputs_):
    # Compute similarity mat of deep feature
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class NCA(nn.Module):
    def __init__(self, alpha=16, k=32, normalized=True):
        super(NCA, self).__init__()
        self.alpha = alpha
        self.K = k
        self.normalized = normalized
        self.use_cuda = False

    def forward(self, inputs, targets):
        if self.normalized:
            inputs = normalize(inputs)
        n = inputs.size(0)
        sim = similarity(inputs)
        eyes_ = Variable(torch.eye(n, n))
        if self.use_cuda:
            targets = targets.cuda()
            eyes_ = eyes_.cuda()

        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim, pos_mask)
        neg_sim = torch.masked_select(sim, neg_mask)

        num_instances = len(pos_sim) // n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.resize(len(pos_sim) // (num_instances - 1), num_instances - 1)
        neg_sim = neg_sim.resize(len(neg_sim) // num_neg_instances, num_neg_instances)

        loss = 0

        for i, pos_pair in enumerate(pos_sim):
            neg_pair = torch.sort(neg_sim[i])[0]
            neg_neig = neg_pair[-self.K:]

            pos_neig = pos_pair

            base = (torch.min(sim[i]) + torch.max(sim[i])) / 2
            pos_logit = torch.sum(torch.exp(self.alpha * (base - pos_neig)))
            neg_logit = torch.sum(torch.exp(self.alpha * (base - neg_neig)))
            loss_ = -torch.log(pos_logit / (pos_logit + neg_logit))

            loss = loss + loss_

        return loss / n

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


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

    loss = NCA(alpha=30)(inputs, targets)
    print(loss)
    loss.backward()


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
