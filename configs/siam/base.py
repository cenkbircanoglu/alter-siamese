from configs.siamese.base import SiameseBaseConfig


class SiamBaseConfig(SiameseBaseConfig):
    def __init__(self):
        super(SiamBaseConfig, self).__init__()
        self.embedding = 128 * 2

    @property
    def network_channel(self):
        return self.channel * 2
