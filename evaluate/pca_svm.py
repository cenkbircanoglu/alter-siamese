# -*- coding: utf-8 -*-
import glob

import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn.utils import shuffle

tr_dir = "/Users/cenk.bircanoglu/personal/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/fashion/train/**/*.jpg"
te_dir = "/Users/cenk.bircanoglu/personal/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/fashion/test/**/*.jpg"
category = {
    "Bluz": 0,
    "Ceket": 1,
    "Elbise": 2,
    "Gömlek": 3,
    "Hırka": 4,
    "Kazak": 5,
    "Mont": 6,
    "Sweatshirt": 7,
    "Tişört": 8,
    "Yelek": 9
}


def read_images(img_dir):
    image_list = []
    labels = []
    for filename in glob.glob(img_dir):
        im = Image.open(filename)
        image_list.append(np.array(im).reshape((3 * 224 * 224)))
        labels.append(category.get(filename.split("/")[-2]))
        im.close()

    return shuffle(np.array(image_list), np.array(labels), random_state=0)


X_tr, y_tr = read_images(tr_dir)

print(X_tr.shape)
pca = IncrementalPCA(n_components=32, batch_size=64)
pca.fit(X_tr)

batch_size = 32
pca_X_tr = None
for i in iter(range((X_tr.shape[0] // batch_size) + 1)):
    print(i * batch_size)
    idx = i * batch_size
    data = X_tr[idx:(idx + batch_size)]
    if data.shape[0] >= 1:
        if type(pca_X_tr) == np.ndarray:
            pca_X_tr = np.concatenate((pca_X_tr, pca.transform(data)))
        else:
            pca_X_tr = pca.transform(data)

np.savetxt("tr_pca.txt", pca_X_tr)
clf = SVC()
clf.fit(pca_X_tr, y_tr)
print("Train Score", clf.score(pca_X_tr, y_tr))
np.savetxt("train_prediction.txt", clf.predict(pca_X_tr))
np.savetxt("train_labels.txt", y_tr)

X_te, y_te = read_images(te_dir)

pca_X_te = None
for i in iter(range((X_te.shape[0] // batch_size) + 1)):
    print(i * batch_size)
    idx = i * batch_size
    data = X_te[idx:(idx + batch_size)]
    if type(pca_X_te) == np.ndarray:
        pca_X_te = np.concatenate((pca_X_te, pca.transform(data)))
    else:
        pca_X_te = pca.transform(data)

np.savetxt("te_pca.txt", pca_X_te)
print('Test Score', clf.score(pca_X_te, y_te))
np.savetxt("test_prediction.txt", clf.predict(pca_X_te))
np.savetxt("test_labels.txt", y_te)
