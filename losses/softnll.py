from torch.nn import _functions


def soft_nll_loss(input, target, weight=None, size_average=True):
    r"""The negative log likelihood loss.
    See :class:`~torch.nn.NLLLoss` for details.
    Args:
        input: :math:`(N, C)` where `C = number of classes` or `(N, C, H, W)` in case of 2D - Loss
        target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.
    Attributes:
        weight: the class-weights given as input to the constructor
    Example:
        >>> # input is of size nBatch x nClasses = 3 x 5
        >>> input = autograd.Variable(torch.randn(3, 5))
        >>> # each element in target has to have 0 <= value < nclasses
        >>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    dim = input.dim()
    if dim == 2:
        f = _functions.thnn.SoftNLLLoss(size_average, weight=weight)
    elif dim == 4:
        f = _functions.thnn.NLLLoss2d(size_average, weight=weight)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))
    return f(input, target)


if __name__ == '__main__':
    from torch.autograd import Variable

    ct = soft_nll_loss()
    # ct = ct.cuda()
    print list(ct.parameters())

    print ct.centers.grad

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct((feat, feat), y)
    out.backward()
    print ct.centers.grad
