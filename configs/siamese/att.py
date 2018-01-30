import os

from configs.base import BaseConfig


class SiameseAtt(BaseConfig):
    def __init__(self):
        super(SiameseAtt, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'att/train/')
        self.te_dir = os.path.join(self.data_dir, 'att/test/')
        self.width = 100
        self.height = 100
        self.channel = 1
        self.network = 'siamese_vgg_100'
