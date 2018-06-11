from torch.nn.modules.loss import BCELoss
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss  # ListWise
from torch.nn.modules.loss import HingeEmbeddingLoss  # Siamese (`1` or `-1`)
from torch.nn.modules.loss import L1Loss
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import MultiLabelSoftMarginLoss  # Listwise
from torch.nn.modules.loss import MultiMarginLoss  # Listwise
from torch.nn.modules.loss import NLLLoss  # ListWise
from torch.nn.modules.loss import NLLLoss2d
from torch.nn.modules.loss import PoissonNLLLoss  # ListWise
from torch.nn.modules.loss import SmoothL1Loss
from torch.nn.modules.loss import SoftMarginLoss  # Listwise

from angle import AngleLoss
from center import CenterLoss
from center2 import CenterLoss2
from contrastive import ContrastiveLoss  # Siamese (`1` or `0`)
from contrastive import ContrastiveLoss  as AllContrastiveLoss  # Siamese (`1` or `0`)
from cosine_embedding import CosineEmbeddingLoss  # Siamese (`1` or `-1`)
from cosine_embedding import CosineEmbeddingLoss as AllCosineEmbeddingLoss  # Siamese (`1` or `-1`)
from focal import FocalLoss
from histogram import HistogramLoss
from margin_ranking import MarginRankingLoss  # Siamese (`1` or `-1`)
from softmax import SoftmaxLoss
from svm import MultiClassHingeLoss
from triplet import TripletMarginLoss
from triplet_2 import TripletMarginLoss2
from contrastive_2 import ContrastiveLoss2
from cosine_embedding_2 import CosineEmbeddingLoss2
__all__ = ['ContrastiveLoss', 'L1Loss', 'NLLLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'NLLLoss2d',
           'CosineEmbeddingLoss', 'HingeEmbeddingLoss', 'MarginRankingLoss',
           'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss', 'CrossEntropyLoss',
           'TripletMarginLoss', 'PoissonNLLLoss', 'CenterLoss', 'CenterLoss2', 'FocalLoss', 'HistogramLoss',
           'MultiClassHingeLoss', 'SoftmaxLoss', 'AngleLoss', 'AllContrastiveLoss', 'AllCosineEmbeddingLoss',
           'TripletMarginLoss2', 'ContrastiveLoss2','CosineEmbeddingLoss2'
           ]
