import random

import matplotlib.pyplot as plt
import numpy as np


def imshow(img, text=None, name=None, should_save=False, i=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if i != None:
        plt.savefig("%s/%s.png" % (name, str(i)))
    else:
        plt.savefig("%s/%s.png" % (name, str(random.random())))


def show_plot(iteration, loss, name):
    plt.plot(iteration, loss)
    plt.savefig("%s/plot_%s.png" % (name, str(random.random())))
