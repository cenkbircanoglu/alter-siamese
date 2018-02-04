#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear SVM with custom multiclass Hinge loss
"""

import torch
import torch.nn as nn


class MultiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(MultiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.weight = weight  # weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average = size_average

    def forward(self, output, y):  # output: batchsize*n_class
        # print(output.requires_grad)
        # print(y.requires_grad)
        output_y = output[torch.arange(0, y.size()[0]).long(), y.data].view(-1, 1)  # view for transpose
        # margin - output[y] + output[i]
        loss = output - output_y + self.margin  # contains i=y
        # remove i=y items
        loss[torch.arange(0, y.size()[0]).long(), y.data] = 0
        # max(0,_)
        #loss[loss < 0] = 0
        # ^p
        if self.p != 1:
            loss = torch.pow(loss, self.p)
        # add weight
        if self.weight is not None:
            loss = loss * self.weight
        # sum up
        loss = torch.sum(loss)
        if self.size_average:
            loss /= output.size()[0]  # output.size()[0]
        return loss
