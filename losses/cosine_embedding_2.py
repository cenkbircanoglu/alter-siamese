import itertools

import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import functional as F


class CosineEmbeddingLoss2(Module):
    r"""Creates a criterion that measures the loss given  an input tensors
    x1, x2 and a `Tensor` label `y` with values 1 or -1.
    This is used for measuring whether two inputs are similar or dissimilar,
    using the cosine distance, and is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    `margin` should be a number from `-1` to `1`, `0` to `0.5` is suggested.
    If `margin` is missing, the default value is `0`.

    The loss function for each sample is::

                     { 1 - cos(x1, x2),              if y ==  1
        loss(x, y) = {
                     { max(0, cos(x1, x2) - margin), if y == -1

    If the internal variable `size_average` is equal to ``True``,
    the loss function averages the loss over the batch samples;
    if `size_average` is ``False``, then the loss function sums over the
    batch samples. By default, `size_average = True`.
    """

    def __init__(self, margin=0.5, size_average=False):
        super(CosineEmbeddingLoss2, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):
        self.indices = list(itertools.combinations([i for i in xrange(target.size(0))], 2))

        first_ind, second_ind = zip(*self.indices)
        first_ind, second_ind = list(first_ind), list(second_ind)
        first_d, second_d = input[first_ind], input[second_ind]
        label = (target[first_ind] == target[second_ind]).type(torch.FloatTensor).cuda()
        label[label == 0] = -1.

        return F.cosine_embedding_loss(first_d, second_d, label, self.margin, self.size_average)



if __name__ == '__main__':
    from torch.autograd import Variable

    ct = CosineEmbeddingLoss2(2)
    # ct = ct.cuda()
    print list(ct.parameters())

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct(feat, y)
    out.backward()
