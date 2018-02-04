import numpy as np
import torch
from torch.autograd import Function

"""
Note:
      1. Here is calculating the log distribution, not the original distribution
      2. For now, not Support Batch: to resolve this problem, we should wrap a loop outside this function
      3. the leverage of numpy shared memory here will not work in cuda tensor: not so nice solution is from the 
        beginning of entrance of the function, check if they are in GPU; transfer them to CPU without exception and 
        to see if we need to transfer them back to GPU according to a flag
      4. convert input to double
"""


class MVNLoss(Function):

    def __init__(self):
        super(MVNLoss, self).__init__()

        self.dim = None
        self.input_minus_miu = None
        self.inv_sing_mat = None
        self.double_flag = None

    def forward(self, miu_vec, cov_mat, input_vec):
        """
        :param params: vector miu and unique elements of covariance matrix sigma
                       the first dim number of params is miu and then is sigma
        :param input: ground truth samples from the training set
        :param dim: the dimension of the distribution
        :return: forward output
        """

        try:
            if isinstance(cov_mat, torch.cuda.DoubleTensor) and isinstance(input_vec, torch.cuda.DoubleTensor) or \
                    isinstance(cov_mat, torch.DoubleTensor) and isinstance(input_vec, torch.DoubleTensor):
                self.double_flag = True
            elif isinstance(cov_mat, torch.cuda.FloatTensor) and isinstance(input_vec, torch.cuda.FloatTensor) or \
                    isinstance(cov_mat, torch.FloatTensor) and isinstance(input_vec, torch.FloatTensor):
                self.double_flag = False
            else:
                raise RuntimeError("params and input have different precisions")
        except RuntimeError as err:
            print(err.message)

        miu_vec = miu_vec.double()
        cov_mat = cov_mat.double()
        input_vec = input_vec.double()

        dim = cov_mat.size(0)

        assert (input_vec.size(1) == dim), "dimension does not match"

        input_minus_miu = input_vec - miu_vec  # 1xdim vector
        inv_sing_mat = torch.inverse(cov_mat)

        self.dim = dim
        self.input_minus_miu = input_minus_miu
        self.inv_sing_mat = inv_sing_mat

        det = np.linalg.det(cov_mat.cpu().numpy())
        try:
            if det < 0:
                raise ValueError("negative determinant of covariance matrix")
        except ValueError as err:
            print(err.message)

        # print('{}\n'.format(det))

        output = - dim / 2.0 * np.log(2 * np.pi) + (-1.0 / 2) * np.log(det) + \
                 (-1.0 / 2) * torch.mm(torch.mm(input_minus_miu, inv_sing_mat), input_minus_miu.t())

        # the output is actually the log pdf
        if not self.double_flag:
            output = output.float()

        return output

    def backward(self, grad_output):
        """
        Here exist two assumptions:
        1. the covariance matrix we construct from the output of RNN is positive definite
        2. the parameters we get from RNN through this operation is [1, ~], while the input is [dim, 1]
        """
        grad_output = grad_output.double()
        grad_output = grad_output[0, 0]  # extract the scalar from PyTorch tensor

        inv_sing_mat = self.inv_sing_mat
        input_minus_miu = self.input_minus_miu
        dim = self.dim

        # gradients of covariance matrix
        mid_result1 = torch.mm(input_minus_miu.t(), input_minus_miu)

        grad_sigma1 = -torch.mm(torch.mm(inv_sing_mat, mid_result1), inv_sing_mat)
        grad_sigma2 = inv_sing_mat
        grad_sigma = -(1.0 / 2) * (grad_sigma1 + grad_sigma2) * grad_output
        # grad_sigma = 2.0*grad_sigma - grad_sigma*torch.eye(dim).double()

        # gradients of miu
        grad_miu = -torch.mm(inv_sing_mat, -input_minus_miu.t()) * grad_output

        # gradients of input
        grad_input = -torch.mm(inv_sing_mat, input_minus_miu.t()) * grad_output

        if not self.double_flag:
            grad_sigma = grad_sigma.float()
            grad_input = grad_input.float()
            grad_miu = grad_miu.float()

        return grad_miu, grad_sigma, grad_input


if __name__ == '__main__':
    from torch.autograd import Variable

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
