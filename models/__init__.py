from models.s100 import siam_net as siam_net_100
from models.s100 import siamese_net as siamese_net_100

from models.s28 import siam_net as siam_net_28
from models.s28 import siamese_net as siamese_net_28
from models.s32 import siam_net as siam_net_32
from models.s32 import siamese_net as siamese_net_32
from models.s64.alexnet import siam_alexnet as siam_alexnet_64
from models.s64.alexnet import siamese_alexnet as siamese_alexnet_64

__all__ = ['siam_alexnet_64', 'siamese_alexnet_64', 'siam_net_28', 'siamese_net_28', 'siam_net_100', 'siamese_net_100',
           'siam_net_32', 'siamese_net_32']
