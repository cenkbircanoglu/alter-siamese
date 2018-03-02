import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CosineLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
            target = target.view(-1, 1)
        dim = 1
        eps = 1e-8
        return F.cosine_similarity(input, target, dim, eps)


if __name__ == '__main__':
    ct = CosineLoss()
    # ct = ct.cuda()
    print list(ct.parameters())

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(8, 4), requires_grad=True)  # .cuda()
    # print feat

    out = ct(feat, y)
    out.backward()
    print ct.centers.grad
