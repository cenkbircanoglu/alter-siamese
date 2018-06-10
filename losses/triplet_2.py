import torch
import torch.nn.functional as F
import itertools

# Inherit from Function
from torch.autograd import Function


class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class TripletMarginLoss2(torch.nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=True):
        super(TripletMarginLoss2, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, input, target):
        self.input_size = target.size(0)
        self.indices = [(d1[0], d2[0], d3[0]) for (d1, d2, d3) in
                        filter(lambda (x, y, z): x[1] == y[1] != z[1] and x[0] != y[0],
                               itertools.combinations(zip([i for i in xrange(target.size(0))], target.data), 3))]
        anchor_ind, pos_ind, neg_ind = zip(*self.indices)
        anchor_ind, pos_ind, neg_ind = list(anchor_ind), list(pos_ind), list(neg_ind)
        anchor, positive, negative = input[anchor_ind], input[pos_ind], input[neg_ind]
        return F.triplet_margin_loss(anchor, positive, negative, self.margin, self.p, self.eps, self.swap)


def module_hook(module, grad_input, grad_out):
    #print(module.indices)
    print('module hook')
    print('grad input')
    print(grad_input)
    print('grad out')
    print(grad_out)
    tensor = torch.autograd.Variable(torch.zeros(module.input_size, 1).float())
    #return tensor,
    for i,(indice1, indice2,indice3) in enumerate(module.indices):
        tensor.data[indice1] = tensor.data[indice1] + grad_input[0].data[i]
        tensor.data[indice2] = tensor.data[indice2] + grad_input[0].data[i]
        tensor.data[indice3] = tensor.data[indice3] + grad_input[0].data[i]

    print('grad_out', tensor)
    return tensor,