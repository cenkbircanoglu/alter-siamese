import torch
import torch.nn as nn

from mvn import MVNLoss


# every row of params is a single sample, and every column of input is a single sample

def mvn_op(cov_chunk, miu_chunk, input):
    num_batch = len(cov_chunk)
    dim = cov_chunk[0].size(0)
    if cov_chunk[0].is_cuda:
        output = Variable(torch.DoubleTensor(num_batch, 1).cuda())
    else:
        output = Variable(torch.DoubleTensor(num_batch, 1))

    for i in xrange(num_batch):
        mvn = MVNLoss()
        cov_temp = cov_chunk[i, :, :]
        cov_temp = cov_temp + cov_temp.t() - torch.diag(torch.diag(cov_temp))
        cov_temp = torch.mm(cov_temp, cov_temp) + Variable(torch.eye(dim, dim).cuda())
        output[i, 0] = mvn(miu_chunk[i, :].unsqueeze(0), cov_temp, input[i, :].unsqueeze(0))

    return output


class MGLoss(nn.Module):
    def __init__(self):
        super(MGLoss, self).__init__()

    def forward(self, mvn_param, lat_var, bbox_input, dim=4):
        lat_num = lat_var.size(1)
        param_stride = int(dim * (dim + 1) / 2 + dim)
        if torch.cuda.is_available():
            output = Variable(torch.zeros(lat_var.size()).cuda())
        else:
            output = Variable(torch.zeros(lat_var.size()))
        for i in range(lat_num):
            # now in order to make the covariance matrix posotive definite do C*C'
            mvn_param_chunk = mvn_param[:, i * param_stride:(i + 1) * param_stride]
            miu_chunk = mvn_param_chunk[:, :dim]
            cov_flat_chunk = mvn_param_chunk[:, dim:param_stride]

            if torch.cuda.is_available():
                cov_chunk = Variable(torch.zeros((mvn_param.size(0), dim, dim)).cuda())
            else:
                cov_chunk = Variable(torch.zeros((mvn_param.size(0), dim, dim)))

            temp_count = 0
            for j in range(dim):
                cov_chunk[:, j:j + 1, j:] = cov_flat_chunk[:, temp_count:temp_count + dim - j].unsqueeze(1)
                temp_count += dim - j

            output[:, i] = mvn_op(cov_chunk, miu_chunk, bbox_input)

        return -torch.sum(torch.sum(output.exp() * lat_var, 1).log()) / lat_var.size(0)


# def mg_loss(mvn_param, lat_var, bbox_input, dim=4):
#     lat_num = lat_var.size(1)
#     param_stride = int(dim*(dim+1)/2+dim)
#     if torch.cuda.is_available():
#         output = Variable(torch.zeros(lat_var.size()).cuda())
#     else:
#         output = Variable(torch.zeros(lat_var.size()))
#     for i in range(lat_num):
#         # now in order to make the covariance matrix posotive definite do C*C'
#         mvn_param_chunk = mvn_param[:, i*param_stride:(i+1)*param_stride]
#         miu_chunk = mvn_param_chunk[:, :dim]
#         cov_flat_chunk = mvn_param_chunk[:, dim:param_stride]
#
#         if torch.cuda.is_available():
#             cov_chunk = Variable(torch.zeros((mvn_param.size(0), dim, dim)).cuda())
#         else:
#             cov_chunk = Variable(torch.zeros((mvn_param.size(0), dim, dim)))
#
#         temp_count = 0
#         for j in range(dim):
#             cov_chunk[:, j:j+1, j:] = cov_flat_chunk[:, temp_count:temp_count+dim-j].unsqueeze(1)
#             temp_count += dim - j
#
#         output[:, i] = mvn_op(cov_chunk, miu_chunk, bbox_input)
#
#     return -torch.sum(torch.sum(output.exp()*lat_var, 1).log())/lat_var.size(0)


if __name__ == '__main__':
    from torch.autograd import Variable

    ct = MGLoss()
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

    ct = MVNLoss()
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
