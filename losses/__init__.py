from torch.nn.modules.loss import BCELoss
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.modules.loss import CosineEmbeddingLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.loss import HingeEmbeddingLoss
from torch.nn.modules.loss import KLDivLoss
from torch.nn.modules.loss import L1Loss
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import MarginRankingLoss
from torch.nn.modules.loss import MultiLabelMarginLoss
from torch.nn.modules.loss import MultiLabelSoftMarginLoss
from torch.nn.modules.loss import MultiMarginLoss
from torch.nn.modules.loss import NLLLoss
from torch.nn.modules.loss import NLLLoss2d
from torch.nn.modules.loss import PoissonNLLLoss
from torch.nn.modules.loss import SmoothL1Loss
from torch.nn.modules.loss import SoftMarginLoss

from contrastive_loss import ContrastiveLoss
from triplet import TripletMarginLoss

__all__ = ['ContrastiveLoss', 'L1Loss', 'NLLLoss', 'KLDivLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'NLLLoss2d',
           'CosineEmbeddingLoss', 'HingeEmbeddingLoss', 'MarginRankingLoss', 'MultiLabelMarginLoss',
           'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss', 'CrossEntropyLoss',
           'TripletMarginLoss', 'PoissonNLLLoss']
