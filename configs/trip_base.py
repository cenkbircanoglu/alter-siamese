from configs.base import BaseConfig


class TripBaseConfig(BaseConfig):
    def __init__(self):
        super(TripBaseConfig, self).__init__()
        self.embedding = 128 * 3
        self.loss = 'TripletMarginLoss'
        self.trainer = "triplet"

    @property
    def network_channel(self):
        return self.channel * 3
