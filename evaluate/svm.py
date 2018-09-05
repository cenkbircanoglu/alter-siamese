import os
from multiprocessing import Process

import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
import concurrent.futures
from evaluate.confusion_matrix import plot_confusion_matrix

__author__ = 'cenk'


def classify_svm(data_path):
    result_path = '%s/svm_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
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

    clf = svm.SVC(kernel='linear', C=1, max_iter=2000000000, random_state=2)
    clf.fit(tr_embeddings, tr_labels)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    # tr_predictions = clf.predict(tr_embeddings)
    # val_predictions = clf.predict(val_embeddings)
    # te_predictions = clf.predict(te_embeddings)
    #
    # tr_fscore = f1_score(tr_predictions, tr_labels, average="weighted")
    # val_fscore = f1_score(val_predictions, val_labels, average="weighted")
    # te_fscore = f1_score(te_predictions, te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\n' % (
                data_path, tr_score, val_score, te_score))
    # with open(result_path, mode='a') as f:
    #     f.write(
    #         'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
    #             data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))
    #
    # conf_mat = confusion_matrix(te_labels, te_predictions)
    # labels = sorted(list(set(list(te_labels))))
    # plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
    #                       output=data_path, path_name='svm_confusion_matrix', alg='svm')


def classify_voting(data_path):
    result_path = '%s/voting_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    print(data_path)

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

    clf1 = LogisticRegression(random_state=2)
    clf2 = RandomForestClassifier(random_state=2)
    clf3 = GaussianNB()

    clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    clf.fit(tr_embeddings, tr_labels)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    tr_predictions = clf.predict(tr_embeddings)
    val_predictions = clf.predict(val_embeddings)
    te_predictions = clf.predict(te_embeddings)

    tr_fscore = f1_score(tr_predictions, tr_labels, average="weighted")
    val_fscore = f1_score(val_predictions, val_labels, average="weighted")
    te_fscore = f1_score(te_predictions, te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
                data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))

    conf_mat = confusion_matrix(te_labels, te_predictions)
    labels = sorted(list(set(list(te_labels))))
    plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
                          output=data_path, path_name='voting_confusion_matrix', alg='voting')


def classify_et(data_path):
    result_path = '%s/et_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    print(data_path)

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

    clf = ExtraTreesClassifier(random_state=2, n_jobs=-1)
    clf.fit(tr_embeddings, tr_labels)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    tr_predictions = clf.predict(tr_embeddings)
    val_predictions = clf.predict(val_embeddings)
    te_predictions = clf.predict(te_embeddings)

    tr_fscore = f1_score(tr_predictions, tr_labels, average="weighted")
    val_fscore = f1_score(val_predictions, val_labels, average="weighted")
    te_fscore = f1_score(te_predictions, te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
                data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))

    conf_mat = confusion_matrix(te_labels, te_predictions)
    labels = sorted(list(set(list(te_labels))))
    plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
                          output=data_path, path_name='et_confusion_matrix', alg='et')


def classify_gb(data_path):
    result_path = '%s/gb_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    print(data_path)

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

    clf = GradientBoostingClassifier(random_state=2)
    clf.fit(tr_embeddings, tr_labels)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    tr_predictions = clf.predict(tr_embeddings)
    val_predictions = clf.predict(val_embeddings)
    te_predictions = clf.predict(te_embeddings)

    tr_fscore = f1_score(tr_predictions, tr_labels, average="weighted")
    val_fscore = f1_score(val_predictions, val_labels, average="weighted")
    te_fscore = f1_score(te_predictions, te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
                data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))

    conf_mat = confusion_matrix(te_labels, te_predictions)
    labels = sorted(list(set(list(te_labels))))
    plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
                          output=data_path, path_name='gb_confusion_matrix', alg='gb')


def classify_mlp(data_path):
    result_path = '%s/mlp_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    print(data_path)

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

    clf = MLPClassifier(random_state=2, max_iter=200000000, hidden_layer_sizes=(64,))
    clf.fit(tr_embeddings, tr_labels)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    tr_predictions = clf.predict(tr_embeddings)
    val_predictions = clf.predict(val_embeddings)
    te_predictions = clf.predict(te_embeddings)

    tr_fscore = f1_score(tr_predictions, tr_labels, average="weighted")
    val_fscore = f1_score(val_predictions, val_labels, average="weighted")
    te_fscore = f1_score(te_predictions, te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
                data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))

    conf_mat = confusion_matrix(te_labels, te_predictions)
    labels = sorted(list(set(list(te_labels))))
    plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
                          output=data_path, path_name='mlp_confusion_matrix', alg='mlp')


def classify_nb(data_path):
    result_path = '%s/nb_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    print(data_path)

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

    clf = MultinomialNB()
    clf.fit(tr_embeddings, tr_labels)

    tr_score = clf.score(tr_embeddings, tr_labels)
    val_score = clf.score(val_embeddings, val_labels)
    te_score = clf.score(te_embeddings, te_labels)

    tr_predictions = clf.predict(tr_embeddings)
    val_predictions = clf.predict(val_embeddings)
    te_predictions = clf.predict(te_embeddings)

    tr_fscore = f1_score(tr_predictions, tr_labels, average="weighted")
    val_fscore = f1_score(val_predictions, val_labels, average="weighted")
    te_fscore = f1_score(te_predictions, te_labels, average="weighted")
    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\tTrain FScore:%s\tVal FScore:%s\tTest FScore:%s\n' % (
                data_path, tr_score, val_score, te_score, tr_fscore, val_fscore, te_fscore))

    conf_mat = confusion_matrix(te_labels, te_predictions)
    labels = sorted(list(set(list(te_labels))))
    plot_confusion_matrix(conf_mat, classes=labels, normalize=True, title='Normalized confusion matrix',
                          output=data_path, path_name='nb_confusion_matrix', alg='nb')


def classify_name(model_name, folder):
    if model_name == 'nb':
        classify_nb(folder)
    elif model_name == 'gb':
        classify_gb(folder)
    elif model_name == 'et':
        classify_et(folder)
    elif model_name == 'mlp':
        classify_mlp(folder)
    elif model_name == 'voting':
        classify_voting(folder)
    elif model_name == 'svm':
        classify_svm(folder)
    elif model_name == 'dl':
        classify_deep_learning(folder)
        classify_deep_learning2(folder)


def classify_all():
    all_folders = []
    for root, directory, filenames in os.walk('/media/cenk/2TB2/alter_siamese/'):
        if len(filenames) > 0 and 'test_embeddings.csv' in filenames:  # and 'Online' in root:
            all_folders.append(root)
    import random
    from itertools import repeat, izip

    params = []
    #params.extend(list(zip(repeat('nb'), all_folders)))
    # params.extend(list(zip(repeat('gb'), all_folders)))
    # params.extend(list(zip(repeat('et'), all_folders)))
    # params.extend(list(zip(repeat('mlp'), all_folders)))
    # params.extend(list(zip(repeat('voting'), all_folders)))
    params.extend(list(zip(repeat('svm'), all_folders)))
    #params.extend(list(zip(repeat('dl'), all_folders)))
    random.shuffle(params)
    print(len(params))
    counter = 0
    for param in all_folders:
        result_path = '%s/svm_results.txt' % os.path.abspath(
            os.path.join(os.path.dirname(param), os.path.join(os.pardir, os.pardir)))
        if os.path.exists(result_path):
            if not param in open(result_path).read():
                counter += 1
    print(counter)

    num_worker = 8
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
       try:
           executor.map(classify_name, *izip(*params), timeout=60)
       except concurrent.futures._base.TimeoutError:
           print("This took to long...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
       try:
           executor.map(classify_name, *izip(*params), timeout=60 * 5)
       except concurrent.futures._base.TimeoutError:
           print("This took to long...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
       try:
           executor.map(classify_name, *izip(*params), timeout=60 * 15)
       except concurrent.futures._base.TimeoutError:
           print("This took to long...")


    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
       try:
           executor.map(classify_name, *izip(*params))
       except concurrent.futures._base.TimeoutError:
           print("This took to long...")

def classify_deep_learning(data_path):
    result_path = '%s/dl_results.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    print(data_path)
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.callbacks import EarlyStopping
    import keras
    # from keras.backend.tensorflow_backend import set_session
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config))

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

    np.random.seed(7)
    num_classes = len(set(tr_labels))

    tr_labels = keras.utils.to_categorical(tr_labels, num_classes)
    val_labels = keras.utils.to_categorical(val_labels, num_classes)
    te_labels = keras.utils.to_categorical(te_labels, num_classes)

    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=tr_embeddings.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=25)]
    model.fit(tr_embeddings, tr_labels, validation_data=(val_embeddings, val_labels), epochs=150, batch_size=128,
              callbacks=callbacks)

    tr_score = model.evaluate(tr_embeddings, tr_labels, batch_size=128)[1]
    val_score = model.evaluate(val_embeddings, val_labels, batch_size=128)[1]
    te_score = model.evaluate(te_embeddings, te_labels, batch_size=128)[1]

    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\n' % (
                data_path, tr_score, val_score, te_score))


def classify_deep_learning2(data_path):
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.callbacks import EarlyStopping
    import keras
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))

    result_path = '%s/dl_results2.txt' % os.path.abspath(
        os.path.join(os.path.dirname(data_path), os.path.join(os.pardir, os.pardir)))
    if os.path.exists(result_path):
        if data_path in open(result_path).read():
            return True
    print(data_path)

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

    np.random.seed(7)
    num_classes = len(set(tr_labels))

    tr_labels = keras.utils.to_categorical(tr_labels, num_classes)
    val_labels = keras.utils.to_categorical(val_labels, num_classes)
    te_labels = keras.utils.to_categorical(te_labels, num_classes)

    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=tr_embeddings.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    model.fit(tr_embeddings, tr_labels, validation_data=(val_embeddings, val_labels), epochs=150, batch_size=128,
              callbacks=callbacks)

    tr_score = model.evaluate(tr_embeddings, tr_labels, batch_size=128)[1]
    val_score = model.evaluate(val_embeddings, val_labels, batch_size=128)[1]
    te_score = model.evaluate(te_embeddings, te_labels, batch_size=128)[1]

    print("tr_score %s" % tr_score)
    print("val_score %s" % val_score)
    print("te_score %s" % te_score)
    with open(result_path, mode='a') as f:
        f.write(
            'Data Path: %s\tTrain Accuracy:%s\tVal Accuracy:%s\tTest Accuracy:%s\n' % (
                data_path, tr_score, val_score, te_score))


if __name__ == '__main__':
    classify_all()
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--data_path', type=str,
    #                     default='/media/cenk/2TB1/alter_siamese/results/mnist/net_28/OnlineContrastiveLossAllPositivePairSelector')
    # args = parser.parse_args()
    #
    # data_path = args.data_path
    # p1 = Process(target=classify_mlp, args=(data_path,))
    # p2 = Process(target=classify_gb, args=(data_path,))
    # p3 = Process(target=classify_voting, args=(data_path,))
    # p4 = Process(target=classify, args=(data_path,))
    # p5 = Process(target=classify_et, args=(data_path,))
    #
    # data_path = data_path.replace("results", "best_results")
    # p6 = Process(target=classify_mlp, args=(data_path,))
    # p7 = Process(target=classify_gb, args=(data_path,))
    # p8 = Process(target=classify_voting, args=(data_path,))
    # p9 = Process(target=classify, args=(data_path,))
    # p10 = Process(target=classify_et, args=(data_path,))
    #
    # processes = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
    #
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()
