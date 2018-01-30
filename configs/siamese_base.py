from configs.base import BaseConfig


class SiameseBaseConfig(BaseConfig):
    def __init__(self):
        super(SiameseBaseConfig, self).__init__()
        self.loss = 'ContrastiveLoss'
        self.embedding = 128 * 1
        self.trainer = "siamese"
        self.margin = 2.0

