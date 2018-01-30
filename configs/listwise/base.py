from configs.base import BaseConfig


class ListWiseConfig(BaseConfig):
    def __init__(self):
        super(ListWiseConfig, self).__init__()
        self.channel = None
        self.loss = 'CrossEntropyLoss'
        self.embedding = 128 * 1
        self.trainer = "listwise"

    @property
    def network_channel(self):
        return self.channel
