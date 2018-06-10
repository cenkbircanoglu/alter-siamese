import os

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
__author__ = 'cenk'


def classify(data_path):
    print(data_path)
    # fname = "{}/best_train_labels.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_train_embeddings.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_val_labels.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_val_embeddings.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_test_labels.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_test_embeddings.csv".format(data_path)
    # os.remove(fname)
    # return True
    # data_path = data_path.replace("results", "best_results")
    result_path = '%s/results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    model_path = '%s/svm.pkl' % data_path
    tr_predictions_path = '%s/train_predictions.csv' % data_path
    val_predictions_path = '%s/val_predictions.csv' % data_path
    te_predictions_path = '%s/test_predictions.csv' % data_path

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

    clf = svm.SVC(kernel='linear', C=1, max_iter=2000000000, verbose=False)
    clf.fit(tr_embeddings, tr_labels)
    joblib.dump(clf, model_path)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    tr_predictions = clf.predict(tr_embeddings)
    val_predictions = clf.predict(val_embeddings)
    te_predictions = clf.predict(te_embeddings)

    np.savetxt(tr_predictions_path, tr_predictions)
    np.savetxt(val_predictions_path, val_predictions)
    np.savetxt(te_predictions_path, te_predictions)

    tr_fscore = f1_score(clf.predict(tr_embeddings), tr_labels, average="weighted")
    val_fscore = f1_score(clf.predict(val_embeddings), val_labels, average="weighted")
    te_fscore = f1_score(clf.predict(te_embeddings), te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
                data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))

def classify1(data_path):
    print(data_path)
    # fname = "{}/best_train_labels.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_train_embeddings.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_val_labels.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_val_embeddings.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_test_labels.csv".format(data_path)
    # os.remove(fname)
    # fname = "{}/best_test_embeddings.csv".format(data_path)
    # os.remove(fname)
    # return True
    # data_path = data_path.replace("results", "best_results")
    result_path = '%s/mlp_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    model_path = '%s/mlp.pkl' % data_path
    tr_predictions_path = '%s/mlp_train_predictions.csv' % data_path
    val_predictions_path = '%s/ml_val_predictions.csv' % data_path
    te_predictions_path = '%s/mlp_test_predictions.csv' % data_path

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

    clf = MLPClassifier(random_state=2, max_iter=200000000, hidden_layer_sizes=(96, 64, 32))
    clf.fit(tr_embeddings, tr_labels)
    joblib.dump(clf, model_path)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    tr_predictions = clf.predict(tr_embeddings)
    val_predictions = clf.predict(val_embeddings)
    te_predictions = clf.predict(te_embeddings)

    np.savetxt(tr_predictions_path, tr_predictions)
    np.savetxt(val_predictions_path, val_predictions)
    np.savetxt(te_predictions_path, te_predictions)

    tr_fscore = f1_score(clf.predict(tr_embeddings), tr_labels, average="weighted")
    val_fscore = f1_score(clf.predict(val_embeddings), val_labels, average="weighted")
    te_fscore = f1_score(clf.predict(te_embeddings), te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
                data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    classify1(args.data_path)
    #classify(args.data_path)
