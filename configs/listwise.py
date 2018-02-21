from configs.base import BaseConfig


class ListWiseConfig(BaseConfig):
    def __init__(self, width=None, height=None, network=None, channel=1, loss="CrossEntropyLoss", embedding=128 * 1,
                 batch_size=256, epochs=20, num_workers=1, cuda=False, **kwargs):
        super(ListWiseConfig, self).__init__(batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                                             channel=channel, cuda=cuda, loss=loss, network=network,
                                             embedding=embedding, **kwargs)
        self.channel = channel
        self.loss = loss
        self.embedding = embedding
        self.width = width
        self.height = height
        self.channel = channel
        self.network = network

    @property
    def network_channel(self):
        return self.channel
