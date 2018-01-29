import os

from configs.base import BaseConfig


class SiameseCifar(BaseConfig):
    def __init__(self):
        super(SiameseCifar, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'cifar/train/')
        self.te_dir = os.path.join(self.data_dir, 'cifar/test/')
        self.width = 32
        self.heigth = 32
        self.channel = 3
        self.network = 'siamese_vgg_32'
