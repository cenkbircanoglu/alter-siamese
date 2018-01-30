import os

from configs.base import BaseConfig


class ListwiseMnist(BaseConfig):
    def __init__(self):
        super(ListwiseMnist, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'mnist/train/')
        self.te_dir = os.path.join(self.data_dir, 'mnist/test/')
        self.width = 28
        self.heigth = 28
        self.channel = 1
        self.network = 'vgg_28'
