import torch
import torch.nn as nn

from config import get_config


class CenterLoss2(nn.Module):
    def __init__(self, lambda_c=1.0, use_cuda=True):
        super(CenterLoss2, self).__init__()
        self.dim_hidden = get_config().embedding
        self.num_classes = get_config().embedding
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.dim_hidden))
        self.use_cuda = use_cuda

    def forward(self, hidden, y):
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.index_select(dim=0, index=y)
        intra_distances = hidden.dist(expanded_centers)
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


if __name__ == '__main__':
    from torch.autograd import Variable

    ct = CenterLoss2(10, 2)
    # ct = ct.cuda()
    print(list(ct.parameters()))

    print(ct.centers.grad)

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct(feat, y)
    out.backward()
    print(ct.centers.grad)
