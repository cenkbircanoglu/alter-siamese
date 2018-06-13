from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class DistWeightLoss(nn.Module):
    def __init__(self, margin=0.02):
        super(DistWeightLoss, self).__init__()
        self.margin = margin
        self.use_cuda = False

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = similarity(inputs)
        eyes_ = Variable(torch.eye(n, n))
        if self.use_cuda:
            targets = targets.cuda()
            eyes_ = eyes_.cuda()

        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim) // n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.resize(len(pos_sim) // (num_instances - 1), num_instances - 1)
        neg_sim = neg_sim.resize(len(neg_sim) // num_neg_instances, num_neg_instances)

        loss = list()
        err = 0

        for i, pos_pair in enumerate(pos_sim):
            pos_pair = torch.sort(pos_pair)[0]
            sampled_index = torch.multinomial(torch.exp(5 * pos_pair), 1)
            neg_pair = torch.sort(neg_sim[i])[0]
            pos_min = pos_pair[sampled_index]
            neg_pair = torch.masked_select(neg_pair, neg_pair > pos_min - 0.01)
            if len(neg_pair) > 0:
                loss.append(torch.mean(neg_pair) - pos_min + 0.01)
                err += 1

        return 0.0 * (torch.mean(pos_min)) if len(loss) == 0 else torch.sum(torch.cat(loss)) / n

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
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(DistWeightLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
