import os

PAR = os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), '../'))


class BaseConfig(object):
    def __init__(self):
        self.data_dir = os.path.join(PAR, 'data')
        self.tr_batch_size = 128
        self.te_batch_size = 1
        self.epochs = 500
        self.margin = 2.0
        self.num_workers = 8
        self.channel = None
        self.result_dir = os.path.join(PAR, './results/%s' % self.__class__.__name__.lower())
        self.log_path = os.path.join(PAR, './results/%s.log' % self.__class__.__name__.lower())
        self.cuda = True
        self.loss = 'ContrastiveLoss'

    @property
    def network_channel(self):
        return self.channel
