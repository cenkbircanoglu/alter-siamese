from torch.nn import Module
from torch.nn import functional as F


class MarginRankingLoss(Module):
    r"""Creates a criterion that measures the loss given
    inputs `x1`, `x2`, two 1D mini-batch `Tensor`s,
    and a label 1D mini-batch tensor `y` with values (`1` or `-1`).

    If `y == 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for `y == -1`.

    The loss function for each sample in the mini-batch is::

        loss(x, y) = max(0, -y * (x1 - x2) + margin)

    if the internal variable `size_average = True`,
    the loss function averages the loss over the batch samples;
    if `size_average = False`, then the loss function sums over the batch
    samples.
    By default, `size_average` equals to ``True``.
    """

    def __init__(self, margin=1, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, (input1, input2), target):
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.size_average)
