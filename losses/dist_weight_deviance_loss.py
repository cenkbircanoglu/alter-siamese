from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


def GaussDistribution(data):
    """
    :param data:
    :return:
    """
    mean_value = torch.mean(data)
    diff = data - mean_value
    std = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return mean_value, std


class DistWeightBinDevianceLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(DistWeightBinDevianceLoss, self).__init__()
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
        c = 0

        for i, pos_pair in enumerate(pos_sim):
            pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_sim[i])[0]

            neg_mean, neg_std = GaussDistribution(neg_pair)
            prob = torch.exp(torch.pow(neg_pair - neg_mean, 2) / (2 * torch.pow(neg_std, 2)))
            neg_index = torch.multinomial(prob, num_instances - 1, replacement=False)

            neg_pair = neg_pair[neg_index]

            if len(neg_pair) < 1:
                c += 1
                continue
            if pos_pair[-1].data[0] > neg_pair[-1].data[0] + 0.05:
                c += 1

            neg_pair = torch.sort(neg_pair)[0]

            pos_loss = torch.mean(torch.log(1 + torch.exp(-2 * (pos_pair - self.margin))))
            neg_loss = 0.04 * torch.mean(torch.log(1 + torch.exp(50 * (neg_pair - self.margin))))
            loss.append(pos_loss + neg_loss)

        return torch.sum(torch.cat(loss)) / n

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

    print(DistWeightBinDevianceLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
