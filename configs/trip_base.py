from configs.triplet_base import TripletBaseConfig


class TripBaseConfig(TripletBaseConfig):
    def __init__(self):
        super(TripBaseConfig, self).__init__()
        self.embedding = 128 * 3

    @property
    def network_channel(self):
        return self.channel * 3
