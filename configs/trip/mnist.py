import os

from configs.trip.base import TripBaseConfig


class TripMnist(TripBaseConfig):
    def __init__(self):
        super(TripMnist, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'mnist/train/')
        self.te_dir = os.path.join(self.data_dir, 'mnist/test/')
        self.width = 28
        self.height = 28
        self.channel = 1
        self.network = 'trip_net_28'
