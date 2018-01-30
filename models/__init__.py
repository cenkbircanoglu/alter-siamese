from models.s100 import siam_net as siam_net_100
from models.s100 import siamese_net as siamese_net_100
from models.s100 import trip_net as trip_net_100
from models.s100 import triplet_net as triplet_net_100
from models.s100.vgg import net as vgg_100
from models.s100.vgg import siam_net as siam_vgg_100
from models.s100.vgg import siamese_net as siamese_vgg_100
from models.s100.vgg import trip_net as trip_vgg_100
from models.s100.vgg import triplet_net as triplet_vgg_100
from models.s28 import siam_net as siam_net_28
from models.s28 import siamese_net as siamese_net_28
from models.s28 import trip_net as trip_net_28
from models.s28 import triplet_net as triplet_net_28
from models.s28.vgg import net as vgg_28
from models.s28.vgg import siam_net as siam_vgg_28
from models.s28.vgg import siamese_net as siamese_vgg_28
from models.s28.vgg import trip_net as trip_vgg_28
from models.s28.vgg import triplet_net as triplet_vgg_28
from models.s32 import siam_net as siam_net_32
from models.s32 import siamese_net as siamese_net_32
from models.s32.vgg import net as vgg_32
from models.s32.vgg import siam_net as siam_vgg_32
from models.s32.vgg import siamese_net as siamese_vgg_32
from models.s32.vgg import trip_net as trip_vgg_32
from models.s32.vgg import triplet_net as triplet_vgg_32
from models.s64.alexnet import siam_alexnet as siam_alexnet_64
from models.s64.alexnet import siamese_alexnet as siamese_alexnet_64

__all__ = ['siam_alexnet_64', 'siamese_alexnet_64', 'siam_net_28', 'siamese_net_28', 'siam_net_100', 'siamese_net_100',
           'siam_net_32', 'siamese_net_32', 'siam_vgg_32', 'siamese_vgg_32', 'siam_vgg_100', 'siamese_vgg_100',
           'siam_vgg_28', 'siamese_vgg_28', 'trip_net_28', 'triplet_net_28', 'trip_vgg_28', 'triplet_vgg_28',
           'trip_vgg_32', 'trip_vgg_100', 'triplet_vgg_32', 'triplet_vgg_100', 'trip_net_100', 'triplet_net_100',
           'vgg_28', 'vgg_32', 'vgg_100']
