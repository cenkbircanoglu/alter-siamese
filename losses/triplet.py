import torch
import torch.nn.functional as F


class TripletMarginLoss(torch.nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=True):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, (anchor, positive, negative)):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin,
                                     self.p, self.eps, self.swap)
