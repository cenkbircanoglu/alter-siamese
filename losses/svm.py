#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear SVM with custom multiclass Hinge loss
"""

import torch
import torch.nn as nn

"""SVM loss
Weston and Watkins version multiclass hinge loss @ https://en.wikipedia.org/wiki/Hinge_loss
for each sample, given output (a vector of n_class values) and label y (an int \in [0,n_class-1])
loss = sum_i(max(0, (margin - output[y] + output[i]))^p) where i=0 to n_class-1 and i!=y

Note: hinge loss is not differentiable
      Let's denote hinge loss as h(x)=max(0,1-x). h'(x) does not exist when x=1, 
      because the left and right limits do not converge to the same number, i.e.,
      h'(1-delta)=-1 but h'(1+delta)=0.

      To overcome this obstacle, people proposed squared hinge loss h2(x)=max(0,1-x)^2. In this case,
      h2'(1-delta)=h2'(1+delta)=0
"""


class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.weight = weight  # weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average = size_average

    def forward(self, output, y):  # output: batchsize*n_class
        # print(output.requires_grad)
        # print(y.requires_grad)
        output_y = output[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()].view(-1, 1)  # view for transpose
        # margin - output[y] + output[i]
        loss = output - output_y + self.margin  # contains i=y
        # remove i=y items
        loss[torch.arange(0, y.size()[0]).long().cuda(), y.data.cuda()] = 0
        # max(0,_)
        loss[loss < 0] = 0
        # ^p
        if (self.p != 1):
            loss = torch.pow(loss, self.p)
        # add weight
        if (self.weight is not None):
            loss = loss * self.weight
        # sum up
        loss = torch.sum(loss)
        if (self.size_average):
            loss /= output.size()[0]  # output.size()[0]
        return loss


if __name__ == '__main__':
    from torch.autograd import Variable

    ct = multiClassHingeLoss()
    # ct = ct.cuda()
    print list(ct.parameters())

    print ct.centers.grad

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct((feat, feat), y)
    out.backward()
    print ct.centers.grad
