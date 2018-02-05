import os

import numpy as np
from sklearn import svm

__author__ = 'cenk'


def classify(data_path):
    fname = "{}/train_labels.csv".format(data_path)
    tr_labels = np.loadtxt(fname)

    fname = "{}/train_embeddings.csv".format(data_path)
    tr_embeddings = np.loadtxt(fname)

    fname = "{}/test_labels.csv".format(data_path)
    te_labels = np.loadtxt(fname)

    fname = "{}/test_embeddings.csv".format(data_path)
    te_embeddings = np.loadtxt(fname)
    print(tr_embeddings.shape)
    clf = svm.SVC(kernel='linear', C=1, max_iter=200000000)
    clf.fit(tr_embeddings, tr_labels)

    tr_score = clf.score(tr_embeddings, tr_labels)

    te_score = clf.score(te_embeddings, te_labels)
    print("tr_score %s" % tr_score)
    print("te_score %s" % te_score)
    with open('%s/results.txt' % os.path.abspath(os.path.join(os.path.dirname(data_path), os.pardir)), mode='a') as f:
        f.write('Data Path: %s\tTrain Accuracy:%s\tTest Accuracy:%s\n' % (data_path, tr_score, te_score))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    classify(args.data_path)
