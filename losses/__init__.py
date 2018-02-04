from torch.nn.modules.loss import BCELoss
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss  # ListWise
from torch.nn.modules.loss import HingeEmbeddingLoss  # Siamese (`1` or `-1`)
from torch.nn.modules.loss import KLDivLoss  # ListWise
from torch.nn.modules.loss import L1Loss
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import MarginRankingLoss  # Siamese (`1` or `-1`)
from torch.nn.modules.loss import MultiLabelMarginLoss  # Listwise
from torch.nn.modules.loss import MultiLabelSoftMarginLoss  # Listwise
from torch.nn.modules.loss import MultiMarginLoss  # Listwise
from torch.nn.modules.loss import NLLLoss  # ListWise
from torch.nn.modules.loss import NLLLoss2d
from torch.nn.modules.loss import PoissonNLLLoss  # ListWise
from torch.nn.modules.loss import SmoothL1Loss
from torch.nn.modules.loss import SoftMarginLoss  # Listwise

from center import CenterLoss
from center2 import CenterLoss2
from contrastive import ContrastiveLoss  # Siamese (`1` or `0`)
from cosine_embedding import CosineEmbeddingLoss  # Siamese (`1` or `-1`)
from focal import FocalLoss
from histogram import HistogramLoss
from lsoftmax import LSoftmaxLinear
from mix_gaussian import MGLoss
from mvn import MVNLoss
from neg import NEG_loss
from softmax import SoftmaxLoss
from softnll import soft_nll_loss
from svm import multiClassHingeLoss
from triplet import TripletMarginLoss
from tsne import tsne_loss

__all__ = ['ContrastiveLoss', 'L1Loss', 'NLLLoss', 'KLDivLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'NLLLoss2d',
           'CosineEmbeddingLoss', 'HingeEmbeddingLoss', 'MarginRankingLoss', 'MultiLabelMarginLoss',
           'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss', 'CrossEntropyLoss',
           'TripletMarginLoss', 'PoissonNLLLoss', 'CenterLoss', 'CenterLoss2', 'FocalLoss', 'HistogramLoss',
           'LSoftmaxLinear', 'soft_nll_loss', 'multiClassHingeLoss', 'tsne_loss', 'NEG_loss', 'MVNLoss',
           'MGLoss', 'SoftmaxLoss']
