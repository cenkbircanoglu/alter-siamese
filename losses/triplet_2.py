import itertools

import torch
import torch.nn.functional as F


class TripletMarginLoss2(torch.nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=True):
        super(TripletMarginLoss2, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, input, target):
        self.input_size = target.size(0)
        self.indices = [(d1[0], d2[0], d3[0]) for (d1, d2, d3) in
                        filter(lambda (x, y, z): x[1] == y[1] != z[1] and x[0] != y[0],
                               itertools.combinations(zip([i for i in xrange(target.size(0))], target.data), 3))]
        anchor_ind, pos_ind, neg_ind = zip(*self.indices)
        anchor_ind, pos_ind, neg_ind = list(anchor_ind), list(pos_ind), list(neg_ind)
        anchor, positive, negative = input[anchor_ind], input[pos_ind], input[neg_ind]
        return F.triplet_margin_loss(anchor, positive, negative, self.margin, self.p, self.eps, self.swap)


if __name__ == '__main__':
    from torch.autograd import Variable

    ct = TripletMarginLoss2()
    print list(ct.parameters())

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct(feat, y)
    print(out)

    out.backward()
