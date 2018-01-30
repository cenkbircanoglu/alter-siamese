import os

from configs.siam_base import SiamBaseConfig


class SiamAtt(SiamBaseConfig):
    def __init__(self):
        super(SiamAtt, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'att/train/')
        self.te_dir = os.path.join(self.data_dir, 'att/test/')
        self.width = 100
        self.heigth = 100
        self.channel = 1
        self.network = 'siam_vgg_100'
