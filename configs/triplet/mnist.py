import os

from configs.triplet.base import TripletBaseConfig


class TripletMnist(TripletBaseConfig):
    def __init__(self):
        super(TripletMnist, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'mnist/train/')
        self.te_dir = os.path.join(self.data_dir, 'mnist/test/')
        self.width = 28
        self.height = 28
        self.channel = 1
        self.network = 'triplet_net_28'
