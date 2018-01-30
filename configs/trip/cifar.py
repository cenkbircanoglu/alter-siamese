import os

from configs.trip.base import TripBaseConfig


class TripCifar(TripBaseConfig):
    def __init__(self):
        super(TripCifar, self).__init__()
        self.tr_dir = os.path.join(self.data_dir, 'cifar/train/')
        self.te_dir = os.path.join(self.data_dir, 'cifar/test/')
        self.width = 32
        self.height = 32
        self.channel = 3
        self.network = 'trip_vgg_32'
