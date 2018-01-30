import os

from configs.listwise.base import ListWiseConfig


class ListWiseMnist(ListWiseConfig):
    def __init__(self):
        super(ListWiseMnist, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'mnist/train/')
        self.te_dir = os.path.join(self.data_dir, 'mnist/test/')
        self.width = 28
        self.height = 28
        self.channel = 1
        self.network = 'vgg_28'
