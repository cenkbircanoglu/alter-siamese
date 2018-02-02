import os

from configs.listwise.base import ListWiseConfig


class ListWiseMini(ListWiseConfig):
    def __init__(self):
        super(ListWiseMini, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'mini/train/')
        self.te_dir = os.path.join(self.data_dir, 'mini/test/')
        self.width = 28
        self.height = 28
        self.channel = 1
        self.network = 'vgg_28'
