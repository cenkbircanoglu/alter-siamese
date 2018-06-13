from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable


# compute the distance matrix of outputs
def euclidean_distances(X):
    batch_size = X.size(0)
    # XX the row norm of X, so is YY
    XX = torch.pow(torch.norm(X, dim=1), 2)
    XX = XX.repeat(batch_size, 1)

    distances = X.mm(X.t())
    distances *= -2
    distances += XX
    distances += XX.t()

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    I_mat = torch.eye(batch_size)
    mask = I_mat.ge(0.5)
    distances = distances.masked_fill(Variable(mask), 0)

    distances = torch.clamp(distances, min=0)
    distances = torch.sqrt(distances)
    return distances


def compute_ID_mat(label):
    size_ = label.size(0)
    label_mat = label.repeat(size_, 1)
    mask_ = label_mat == label_mat.t()
    # change datatype form byte to int
    ID_mat = Variable(torch.zeros(size_, size_))
    ID_mat = ID_mat.masked_fill(mask_, 1)
    return ID_mat


class BatchHardLoss:

    def __init__(self, margin=1):
        self.margin = margin

    def forward(self, outputs, labels):
        batch_size = outputs.size(0)
        dist_mat = euclidean_distances(outputs)
        # ID_mat here is ByteTensor,  can not made into any computation

        ID_mat = compute_ID_mat(labels.clone())

        pos_dist_mat = Variable(torch.zeros(batch_size, batch_size))
        pos_dist_mat = torch.addcmul(pos_dist_mat, 1, ID_mat, dist_mat)

        neg_dist_mat = Variable(torch.zeros(batch_size, batch_size))
        neg_dist_mat = torch.addcmul(neg_dist_mat, 1, 1 - ID_mat, dist_mat)
        mask_ = (neg_dist_mat == 0)
        neg_dist_mat.masked_fill_(mask_, 10000)

        hard_pos = torch.max(pos_dist_mat, dim=0)[0]
        hard_neg = torch.min(neg_dist_mat, dim=0)[0]

        triplet_losses = torch.clamp(hard_pos - hard_neg + self.margin, min=0)

        return torch.sum(triplet_losses)

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


def main():
    data_size = 4
    input_dim = 3
    output_dim = 2
    num_class = 2
    margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    print('training data is ', x)
    print('initial parameters are ', w)
    out = x.mm(w)
    print('extracted feature is :', out)

    y_ = np.random.randint(num_class, size=data_size)
    y = Variable(torch.from_numpy(y_))
    BH = BatchHardLoss(margin=margin)
    loss = BH.forward(out, y)
    print('loss is :', loss)
    loss.backward()
    print('grad is: ', w.grad)


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
