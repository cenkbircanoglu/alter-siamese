import torch
import torch.nn.functional as F
import itertools


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
    print(module.indices)
    print('module hook')
    print('grad input')
    print(grad_input)
    print('grad out')
    print(grad_out)
    tensor = torch.autograd.Variable(torch.zeros(module.input_size, 1).float())
    #return tensor,
    for i,(indice1, indice2,indice3) in enumerate(module.indices):
        tensor[indice1].add(-1,grad_input[0][i])
        tensor[indice2].add(-1,grad_input[0][i])
        tensor[indice3].add(-1,grad_input[0][i])
        print(tensor)
    print('grad_out', tensor)
