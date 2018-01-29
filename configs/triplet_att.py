import os

from configs.base import BaseConfig


class TripletAtt(BaseConfig):
    def __init__(self):
        super(TripletAtt, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'att/train/')
        self.te_dir = os.path.join(self.data_dir, 'att/test/')
        self.width = 100
        self.heigth = 100
        self.channel = 1
        self.network = 'triplet_vgg_100'
