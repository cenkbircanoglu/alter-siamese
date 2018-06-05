import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, (output1, output2), label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        label = label.type(torch.FloatTensor).cuda()

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


if __name__ == '__main__':
    from torch.autograd import Variable

    ct = ContrastiveLoss(2)
    # ct = ct.cuda()
    print list(ct.parameters())

    print ct.centers.grad

    y = Variable(torch.Tensor([0, 0, 0, 1]))  # .cuda()
    # print y
    feat = Variable(torch.zeros(4, 2), requires_grad=True)  # .cuda()
    # print feat

    out = ct(feat, y)
    out.backward()
    print ct.centers.grad
