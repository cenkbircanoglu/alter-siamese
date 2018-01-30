import os

from configs.listwise.base import ListWiseConfig


class ListWiseCifar(ListWiseConfig):
    def __init__(self):
        super(ListWiseCifar, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'cifar/train/')
        self.te_dir = os.path.join(self.data_dir, 'cifar/test/')
        self.width = 32
        self.height = 32
        self.channel = 3
        self.network = 'vgg_32'
