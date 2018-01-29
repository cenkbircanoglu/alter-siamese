import os

from configs.trip_base import TripBaseConfig


class TripMnist(TripBaseConfig):
    def __init__(self):
        super(TripMnist, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'mnist/train/')
        self.te_dir = os.path.join(self.data_dir, 'mnist/test/')
        self.width = 28
        self.heigth = 28
        self.channel = 1
        self.network = 'trip_net_28'
