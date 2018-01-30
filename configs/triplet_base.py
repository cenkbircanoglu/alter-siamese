from configs.base import BaseConfig


class TripletBaseConfig(BaseConfig):
    def __init__(self):
        super(TripletBaseConfig, self).__init__()
        self.loss = 'TripletMarginLoss'
        self.trainer = "triplet"

