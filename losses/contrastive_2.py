import itertools

import torch
import torch.nn.functional as F


class ContrastiveLoss2(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss2, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        self.indices = list(itertools.combinations([i for i in xrange(target.size(0))], 2))

        first_ind, second_ind = zip(*self.indices)
        first_ind, second_ind = list(first_ind), list(second_ind)
        first_d, second_d = input[first_ind], input[second_ind]
        label = target[first_ind] == target[second_ind]

        euclidean_distance = F.pairwise_distance(first_d, second_d)
        label = label.type(torch.FloatTensor).cuda()

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


if __name__ == '__main__':
    from torch.autograd import Variable

    ct = ContrastiveLoss2(2)
    # ct = ct.cuda()
    print list(ct.parameters())

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct(feat, y)
    out.backward()
