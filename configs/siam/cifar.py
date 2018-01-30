import os

from configs.siam.base import SiamBaseConfig


class SiamCifar(SiamBaseConfig):
    def __init__(self):
        super(SiamCifar, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'cifar/train/')
        self.te_dir = os.path.join(self.data_dir, 'cifar/test/')
        self.width = 32
        self.height = 32
        self.channel = 3
        self.network = 'siam_vgg_32'
