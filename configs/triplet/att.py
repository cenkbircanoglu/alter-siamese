import os

from configs.triplet.base import TripletBaseConfig


class TripletAtt(TripletBaseConfig):
    def __init__(self):
        super(TripletAtt, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'att/train/')
        self.te_dir = os.path.join(self.data_dir, 'att/test/')
        self.width = 100
        self.height = 100
        self.channel = 1
        self.network = 'triplet_net_100'
