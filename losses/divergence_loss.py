from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

"""
Divergence Loss in ABE

Attention-based Ensemble for Deep Metric Learning

"""


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


# TODO
class DivergenceLoss(nn.Module):
    def __init__(self, num_classifier=8, Embed_dim=64, margin=0.2, alpha=20):
        super(DivergenceLoss, self).__init__()
        self.num_classifier = num_classifier
        self.Embed_dim = Embed_dim
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = normalize(inputs)
        n = inputs.size(0)

        loss = 0
        for input_ in inputs:
            input_ = input_.view(self.num_classifier, -1)
            input_ = normalize(input_)

            sim_mat = similarity(input_)
            loss += torch.mean(torch.clamp(sim_mat, min=self.margin)) - self.margin

        return loss / n

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))

    print(DivergenceLoss()(inputs))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
