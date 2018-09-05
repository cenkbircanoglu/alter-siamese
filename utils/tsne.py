#!/usr/bin/env python2

import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import concurrent.futures

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import repeat, izip

plt.style.use('bmh')
import os


def tsne(work_dir, dtype):
    out = "{}/{}_tsne.pdf".format(work_dir, dtype)
    print(work_dir)
    if os.path.exists(out):
        print(out)
        return True
    y = np.loadtxt("{}/{}_labels.csv".format(work_dir, dtype))
    X = np.loadtxt("{}/{}_embeddings.csv".format(work_dir, dtype))

    target_names = np.array(list(set(y)))
    colors = cm.Dark2(np.linspace(0, 1, len(target_names)))

    if X.shape[1] < 50:
        X_pca = PCA(n_components=int(X.shape[1] / 2)).fit_transform(X, X)
    else:
        X_pca = PCA(n_components=50, random_state=0).fit_transform(X, X)
    tsne = TSNE(n_components=2, init='random', random_state=0, verbose=0)
    X_r = tsne.fit_transform(X_pca)

    for c, i, target_name in zip(colors, list(range(1, len(target_names) + 1)), target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.legend()

    plt.savefig(out)
    print("Saved to: {}".format(out))


if __name__ == '__main__':
    all_folders = []
    dtypes = ['train', 'test', 'val']
    for root, directory, filenames in os.walk('/media/cenk/2TB2/alter_siamese/results/mnist'):
        if len(filenames) > 0 and 'train_embeddings.csv' in filenames:
            all_folders.extend(list(zip(repeat(root), dtypes)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(tsne, *izip(*all_folders))
