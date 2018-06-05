from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn as nn


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, input):
        input = input.squeeze()
        return input.div(torch.norm(input, dim=1).view(-1, 1))

    def __repr__(self):
        return self.__class__.__name__


class HistogramLoss(torch.nn.Module):
    def __init__(self, num_steps=150, cuda=True):
        super(HistogramLoss, self).__init__()
        from config import get_config
        #num_steps = get_config().embedding
        self.step = 2 / (num_steps - 1)
        self.use_cuda = False
        self.t = torch.range(-1, 1, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        self.l2 = L2Normalization()

    def forward(self, features, classes):
        features = self.l2.forward(features)

        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (delta_repeat == (self.t - self.step)) & inds
            indsb = (delta_repeat == self.t) & inds
            s_repeat_[~(indsb | indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - Variable(self.t) + self.step)[indsa] / self.step
            s_repeat_[indsb] = (-s_repeat_ + Variable(self.t) + self.step)[indsb] / self.step

            return s_repeat_.sum(1) / size

        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1) == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = torch.mm(features, features.transpose(0, 1))
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        if self.use_cuda:
            s_inds = s_inds.cuda()
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum()
        neg_size = (~classes_eq[s_inds]).sum()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        delta_repeat = (torch.floor((s_repeat.data + 1) / self.step) * self.step - 1).float()

        histogram_pos = histogram(pos_inds, pos_size)
        histogram_neg = histogram(neg_inds, neg_size)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
        if self.use_cuda:
            histogram_pos_inds = histogram_pos_inds.cuda()
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
    
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        self.t = self.t.cuda()
        return self._apply(lambda t: t.cuda(device_id))


if __name__ == '__main__':
    ct = HistogramLoss()
    # ct = ct.cuda()
    print list(ct.parameters())

    y = Variable(torch.FloatTensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.rand(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct(feat, y)
    out.backward()
    print ct.centers.grad
