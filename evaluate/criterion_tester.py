import os

import numpy as np
import torch
from torch.autograd import Variable

__author__ = 'cenk'


def classify(data_path):
    print(data_path)
    result_path = '%s/results.txt' % os.path.abspath(os.path.join(os.path.dirname(data_path), os.pardir))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    fname = "{}/train_labels.csv".format(data_path)
    if not os.path.exists(fname):
        return True

    tr_labels = np.loadtxt(fname)

    fname = "{}/train_embeddings.csv".format(data_path)
    tr_embeddings = np.loadtxt(fname)

    fname = "{}/val_labels.csv".format(data_path)
    val_labels = np.loadtxt(fname)

    fname = "{}/val_embeddings.csv".format(data_path)
    val_embeddings = np.loadtxt(fname)

    fname = "{}/test_labels.csv".format(data_path)
    te_labels = np.loadtxt(fname)

    fname = "{}/test_embeddings.csv".format(data_path)
    te_embeddings = np.loadtxt(fname)

    print(tr_embeddings.shape)
    tr_count = float(tr_embeddings.shape[0])
    val_count = float(val_embeddings.shape[0])
    te_count = float(te_embeddings.shape[0])

    tr_embeddings = Variable(torch.Tensor(tr_embeddings))
    val_embeddings = Variable(torch.Tensor(val_embeddings))
    te_embeddings = Variable(torch.Tensor(te_embeddings))
    tr_labels = Variable(torch.LongTensor(tr_labels))
    val_labels = Variable(torch.LongTensor(val_labels))
    te_labels = Variable(torch.LongTensor(te_labels))

    tr_score = np.sum(tr_embeddings.data.max(1, keepdim=True)[1].numpy().T == tr_labels.data.numpy())
    val_score = np.sum(val_embeddings.data.max(1, keepdim=True)[1].numpy().T == val_labels.data.numpy())
    te_score = np.sum(te_embeddings.data.max(1, keepdim=True)[1].numpy().T == te_labels.data.numpy())

    print(tr_score / tr_count, val_score / val_count, te_score / te_count)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    classify(args.data_path)
