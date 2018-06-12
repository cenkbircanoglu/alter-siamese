import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin=1., triplet_selector=None, p=2, eps=1e-6, swap=True):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        anchor = embeddings[triplets[:, 0]]
        positive = embeddings[triplets[:, 1]]
        negative = embeddings[triplets[:, 2]]
        return F.triplet_margin_loss(anchor, positive, negative, self.margin, self.p, self.eps, self.swap)
