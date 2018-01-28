import matplotlib.pyplot as plt
import numpy as np

from config import Config


def imshow(img, text=None, name=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(0, 0, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("%s/%s.png" % (Config.result_dir, str(name)))


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig("%s/loss_plot.png" % (Config.result_dir))
