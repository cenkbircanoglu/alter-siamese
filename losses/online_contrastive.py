import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin=2.0, pair_selector=None):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        output1 = torch.cat([embeddings[positive_pairs[:, 0]], embeddings[negative_pairs[:, 0]]], dim=0).cuda()
        output2 = torch.cat([embeddings[positive_pairs[:, 1]], embeddings[negative_pairs[:, 1]]], dim=0).cuda()
        label = Variable(
            torch.cat([torch.ones(positive_pairs.shape[0]), torch.zeros(negative_pairs.shape[0])], dim=0).cuda())

        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
