from __future__ import print_function, absolute_import

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


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class CenterTripletLoss(nn.Module):
    def __init__(self):
        super(CenterTripletLoss, self).__init__()

        self.use_cuda = False

    def forward(self, inputs, targets):
        n = inputs.size(0)
        num_dim = inputs.size(1)
        targets_ = list(set(targets.data))
        num_class = len(targets_)

        targets_ = Variable(torch.IntTensor(targets_))

        mask_ = targets.repeat(num_class, 1).eq(targets_.repeat(n, 1).t())
        _mask = Variable(torch.ByteTensor(num_class, n).fill_(1)) - mask_
        if self.use_cuda:
            targets_.cuda()
            _mask.cuda()

        centers = []
        inputs_list = []

        for i, target in enumerate(targets_):
            mask_i = mask_[i].repeat(num_dim, 1).t()
            input_ = inputs[mask_i].resize(len(inputs[mask_i]) // num_dim, num_dim)
            centers.append(torch.mean(input_, 0))
            inputs_list.append(input_)

        centers = [centers[i].resize(1, num_dim) for i in range(len(centers))]
        centers = torch.cat(centers, 0)

        centers_dist = pair_euclidean_dist(centers, inputs)
        neg_dist = centers_dist[_mask].resize(num_class - 1, n)
        pos_dist = centers_dist[mask_]

        return torch.mean(pos_dist.clamp(min=0.15) - torch.log(torch.sum(torch.exp(-neg_dist.clamp(max=0.6)), 0)))

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

    print(CenterTripletLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
