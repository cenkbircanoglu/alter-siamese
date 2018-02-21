from configs.base import BaseConfig


class TripletConfig(BaseConfig):
    def __init__(self, width=None, height=None, network=None, channel=1, loss="TripletMarginLoss", embedding=128 * 1,
                 batch_size=128, epochs=20, num_workers=1, cuda=False, **kwargs):
        super(TripletConfig, self).__init__(batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                                            channel=channel, cuda=cuda, loss=loss, network=network, embedding=embedding,
                                            **kwargs)
        self.channel = channel
        self.loss = loss
        self.embedding = embedding
        self.width = width
        self.height = height
        self.channel = channel
        self.network = network
        self.margin = 2.0


class TripConfig(TripletConfig):
    def __init__(self, embedding=128 * 3, **kwargs):
        super(TripConfig, self).__init__(**kwargs)
        self.embedding = embedding

    @property
    def network_channel(self):
        return self.channel * 3
