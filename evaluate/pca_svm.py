# -*- coding: utf-8 -*-
import glob

import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm

data_name = 'cats_dogs'
tr_dir = "/media/cenk/2TB1/alter_siamese/data/%s/train/**/*.*" % data_name
val_dir = "/media/cenk/2TB1/alter_siamese/data/%s/val/**/*.*" % data_name
te_dir = "/media/cenk/2TB1/alter_siamese/data/%s/test/**/*.*" % data_name

result_dir = "results/%s/" % data_name
try:
    import os

    os.makedirs(result_dir)
except:
    pass
batch_size = 128
img_dim = 3 * 64 * 64
n_components = 128


def read_images(img_dir):
    image_list = []
    labels = []
    for filename in tqdm(glob.glob(img_dir)):
        if '.jpg' in filename or '.png' in filename:
            im = Image.open(filename)
            image_list.append(np.array(im).reshape((img_dim)))
            labels.append(filename.split("/")[-2])
            im.close()

    return shuffle(np.array(image_list), np.array(labels), random_state=0)


X_tr, y_tr = read_images(tr_dir)
print(X_tr.shape)
enc = LabelEncoder()
y_tr = enc.fit_transform(y_tr)

pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
pca.fit(X_tr, )


def transform(X):
    pca_X = None
    for i in iter(range((X.shape[0] // batch_size) + 1)):
        idx = i * batch_size
        data = X[idx:(idx + batch_size)]
        if data.shape[0] >= 1:
            if type(pca_X) == np.ndarray:
                pca_X = np.concatenate((pca_X, pca.transform(data)))
            else:
                pca_X = pca.transform(data)
    return pca_X


pca_X_tr = transform(X_tr)
np.savetxt("%s/tr_pca.txt" % result_dir, pca_X_tr)
clf = SVC()
clf.fit(pca_X_tr, y_tr)
tr_score = clf.score(pca_X_tr, y_tr)
print("Train Score", tr_score)
np.savetxt("%s/train_prediction.txt" % result_dir, clf.predict(pca_X_tr))
np.savetxt("%s/train_labels.txt" % result_dir, y_tr)

X_val, y_val = read_images(val_dir)
y_val = enc.transform(y_val)
pca_X_val = transform(X_val)

np.savetxt("%s/val_pca.txt" % result_dir, pca_X_val)
val_score = clf.score(pca_X_val, y_val)
print('Val Score', val_score)
np.savetxt("%s/val_prediction.txt" % result_dir, clf.predict(pca_X_val))
np.savetxt("%s/val_labels.txt" % result_dir, y_val)

X_te, y_te = read_images(te_dir)
y_te = enc.transform(y_te)
pca_X_te = transform(X_te)

np.savetxt("%s/te_pca.txt" % result_dir, pca_X_te)
te_score = clf.score(pca_X_te, y_te)
print('Test Score', te_score)
np.savetxt("%s/test_prediction.txt" % result_dir, clf.predict(pca_X_te))
np.savetxt("%s/test_labels.txt" % result_dir, y_te)

print(tr_score, val_score, te_score)

"""
CIFAR10
('Train Score', 1.0)
('Val Score', 0.10425)
('Test Score', 0.1)""
CIFAR100
('Train Score', 0.9998666666666667)
('Val Score', 0.0068)
('Test Score', 0.0104)
MNIST
('Train Score', 1.0)
('Val Score', 0.10883333333333334)
('Test Score', 0.11444444444444445) 
Fashion
('Train Score', 0.999894140687027)
('Val Score', 0.15769414006669843)
('Test Score', 0.17357945068386524)
Marvel
('Train Score', 0.959858790590097)
('Val Score', 0.05806628321007943)
('Test Score', 0.05036431036686693)
Cats_Dogs
('Train Score', 0.9999365039050099)
('Val Score', 0.48027444253859347)
('Test Score', 0.4988)
"""
