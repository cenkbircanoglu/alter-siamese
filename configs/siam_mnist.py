import os

from configs.siam_base import SiamBaseConfig


class SiamMnist(SiamBaseConfig):
    def __init__(self):
        super(SiamMnist, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'mnist/train/')
        self.te_dir = os.path.join(self.data_dir, 'mnist/test/')
        self.width = 28
        self.heigth = 28
        self.channel = 1
        self.network = 'siam_vgg_28'
