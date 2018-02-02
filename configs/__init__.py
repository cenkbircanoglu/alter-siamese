from configs.listwise.att import ListWiseAtt
from configs.listwise.cifar import ListWiseCifar
from configs.listwise.mini import ListWiseMini
from configs.listwise.mnist import ListWiseMnist
from configs.siam.att import SiamAtt
from configs.siam.cifar import SiamCifar
from configs.siam.mnist import SiamMnist
from configs.siamese.att import SiameseAtt
from configs.siamese.cifar import SiameseCifar
from configs.siamese.mnist import SiameseMnist
from configs.trip.att import TripAtt
from configs.trip.cifar import TripCifar
from configs.trip.mnist import TripMnist
from configs.triplet.att import TripletAtt
from configs.triplet.cifar import TripletCifar
from configs.triplet.mnist import TripletMnist

__all__ = ['SiamMnist', 'SiamCifar', 'SiameseAtt', 'SiamAtt', 'SiameseMnist', 'SiameseCifar', 'TripMnist',
           'TripletMnist', 'TripletAtt', 'TripCifar', 'TripAtt', 'TripletCifar', 'ListWiseCifar', 'ListWiseMnist',
           'ListWiseMnist', 'ListWiseMini']
