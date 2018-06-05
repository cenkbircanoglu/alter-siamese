from configs.base import BaseConfig


class SiameseConfig(BaseConfig):
    def __init__(self, width=None, height=None, network=None, channel=1, loss="ContrastiveLoss", embedding=128 * 1,
                 batch_size=128, epochs=20, num_workers=16, cuda=False, negative=0, positive=1, **kwargs):
        super(SiameseConfig, self).__init__(batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                                            channel=channel, cuda=cuda, loss=loss, embedding=embedding, network=network,
                                            **kwargs)
        self.channel = channel
        self.loss = loss
        self.embedding = embedding
        self.width = width
        self.height = height
        self.channel = channel
        self.network = network
        self.margin = 2.0
        self.negative = negative
        self.positive = positive


class SiamConfig(SiameseConfig):
    def __init__(self, embedding=128 * 2, **kwargs):
        super(SiamConfig, self).__init__(**kwargs)
        self.embedding = embedding

    @property
    def network_channel(self):
        return self.channel * 2
