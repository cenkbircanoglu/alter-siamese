import os
import pickle
import random

import numpy as np

from utils.make_dirs import create_dir


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def unpickler(file):
    with open(file, 'rb') as fo:
        unpickler = pickle.Unpickler(fo)
        # if file is not empty scores will be equal
        # to the value unpickled
        dict = unpickler.load()
    return dict
def load_val(data_folder, img_size=64):
    d = unpickler(data_folder + 'val_data')
    x = d['data']
    y = d['labels']

    y = [i - 1 for i in y]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    from scipy import misc
    folder = data_folder + '/test'
    for x1, y1 in zip(x, y):
        create_dir("%s/%s" % (folder, y1))
        misc.imsave('%s/%s/%s.jpg' % (folder, y1, random.random()), x1)


def load_databatch(data_folder, idx, img_size=64):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']

    y = [i - 1 for i in y]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    from scipy import misc
    folder = data_folder + '/train'
    for x1, y1 in zip(x, y):
        create_dir("%s/%s" % (folder, y1))
        misc.imsave('%s/%s/%s.jpg' % (folder, y1, random.random()), x1)


data_folder = "/Users/cenk.bircanoglu/personal/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/imagenet/"
load_val(data_folder)
# load_databatch(data_folder, 1)
# load_databatch(data_folder, 2)
# load_databatch(data_folder, 3)
# load_databatch(data_folder, 4)
# load_databatch(data_folder, 5)
# load_databatch(data_folder, 6)
# load_databatch(data_folder, 7)
# load_databatch(data_folder, 8)
# load_databatch(data_folder, 9)
# load_databatch(data_folder, 10)
