from configs.base import BaseConfig


class SiamBaseConfig(BaseConfig):
    def __init__(self):
        super(SiamBaseConfig, self).__init__()

    @property
    def network_channel(self):
        return self.channel * 2
