import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class OnlineCosineLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin=0.5, pair_selector=None, size_average=False):
        super(OnlineCosineLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        self.size_average = size_average

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        output1 = torch.cat([embeddings[positive_pairs[:, 0]], embeddings[negative_pairs[:, 0]]], dim=0).cuda()
        output2 = torch.cat([embeddings[positive_pairs[:, 1]], embeddings[negative_pairs[:, 1]]], dim=0).cuda()
        label = Variable(
            torch.cat([torch.ones(positive_pairs.shape[0]), torch.zeros(negative_pairs.shape[0])], dim=0).cuda())
        label[label == 0] = -1

        return F.cosine_embedding_loss(output1, output2, label, self.margin, self.size_average)
