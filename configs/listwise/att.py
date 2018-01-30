import os

from configs.listwise.base import ListWiseConfig


class ListWiseAtt(ListWiseConfig):
    def __init__(self):
        super(ListWiseAtt, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'att/train/')
        self.te_dir = os.path.join(self.data_dir, 'att/test/')
        self.width = 100
        self.height = 100
        self.channel = 1
        self.network = 'vgg_100'
