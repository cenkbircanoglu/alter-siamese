from torch.autograd import Variable
from torch.nn import Module
from torch.nn import functional as F


class CosineEmbeddingLoss(Module):
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
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, (input1, input2), target):
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.size_average)


if __name__ == '__main__':
    import torch

    cuda_num = 2
    batch_size = 50

    loss_fn = torch.nn.CosineEmbeddingLoss()

    for i in range(500):
        sentences = Variable(torch.randn(batch_size, 300))
        images = Variable(torch.randn(batch_size, 400))
        flags = Variable(torch.ones(batch_size))

        fc = torch.nn.Linear(400, 300)
        optimizer = torch.optim.SGD(fc.parameters(), lr=1e-4)
        img_emb = fc(images)
        sen_emb = sentences

        loss = loss_fn(img_emb, sen_emb, flags)
        print 'Batch_id %d \t  loss %.2f' % (i, loss.data.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
