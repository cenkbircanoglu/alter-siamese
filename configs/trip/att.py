import os

from configs.trip.base import TripBaseConfig


class TripAtt(TripBaseConfig):
    def __init__(self):
        super(TripAtt, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'att/train/')
        self.te_dir = os.path.join(self.data_dir, 'att/test/')
        self.width = 100
        self.height = 100
        self.channel = 1
        self.network = 'trip_net_100'
