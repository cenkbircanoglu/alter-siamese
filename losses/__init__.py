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

from losses.others.a_batch_all import ABatchAll
from losses.others.a_hard_pair import AHardPair
from losses.others.a_triplet import ATriplet
from losses.angle import AngleLoss
from losses.others.batch_all import BatchAll
from losses.others.batch_hard import BatchHardLoss
from losses.others.bdw_neighbour_loss import BDWNeighbourLoss
from losses.others.bin_branch_loss import BinBranchLoss
from losses.others.bin_deviance_loss import BinDevianceLoss
#from losses.branch_knn_softmax import BranchKNNSoftmax
from losses.center import CenterLoss
from losses.center2 import CenterLoss2
from losses.others.center_loss import CenterLoss as CenterNewLoss
from losses.others.center_nca_loss import CenterNCALoss
from losses.others.center_triplet import CenterTripletLoss
from losses.others.cluster_nca_loss import ClusterNCALoss
from losses.contrastive import ContrastiveLoss  # Siamese (`1` or `0`)
from losses.contrastive import ContrastiveLoss  as AllContrastiveLoss  # Siamese (`1` or `0`)
from losses.contrastive_loss import ContrastiveLoss as ContrastiveNewLoss
from losses.cosine_embedding import CosineEmbeddingLoss  # Siamese (`1` or `-1`)
from losses.cosine_embedding import CosineEmbeddingLoss as AllCosineEmbeddingLoss  # Siamese (`1` or `-1`)
from losses.others.decor_loss import DecorLoss
from losses.others.deviance_loss import DevianceLoss
from losses.others.dist_weight_contrastive_loss import DistWeightContrastiveLoss
from losses.others.dist_weight_dev_branch_loss import DistWeightDevBranchLoss
from losses.others.dist_weight_deviance_loss import DistWeightBinDevianceLoss
from losses.others.dist_weight_loss import DistWeightLoss
from losses.others.dist_weight_neighbour_loss import DistWeightNeighbourLoss
from losses.others.distance_match_loss import DistanceMatchLoss
from losses.others.divergence_loss import DivergenceLoss
from losses.others.ensemble_dw_neighbour_loss import EnsembleDWNeighbourLoss
from losses.focal import FocalLoss
from losses.others.gaussian_lda import GaussianLDA
from losses.others.gaussian_metric import GaussianMetricLoss
from losses.others.grad_nca import Grad_NCA
from losses.others.hdc_bin_loss import HDC_Bin_Loss
from losses.others.hdc_loss import HDCLoss
from losses.histogram import HistogramLoss
from losses.others.histogram_loss import HistogramLoss as HistogramNewLoss
#from losses.js_div_knn_softmax_loss import JSDivKNNSoftmaxLoss
from losses.others.js_div_loss import JSDivLoss
from losses.others.kmean_loss import KmeanLoss
from losses.others.margin_deviance_loss import MarginDevianceLoss
from losses.others.margin_positive_loss import MarginPositiveLoss
from losses.margin_ranking import MarginRankingLoss  # Siamese (`1` or `-1`)
from losses.others.mca_loss import MCALoss
from losses.others.mean_bin_deviance_loss import MeanBinDevianceLoss
from losses.others.nca import NCA
from losses.softmax import SoftmaxLoss
from losses.svm import MultiClassHingeLoss
from losses.triplet import TripletMarginLoss
from losses.others.neighbour_hard_loss import NeighbourHardLoss
from losses.others.neighbour_loss import NeighbourLoss
from losses.others.ori_bin_loss import OriBinLoss
from losses.others.softmax_neig_loss import SoftmaxNeigLoss

__all__ = ['ContrastiveLoss', 'L1Loss', 'NLLLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'NLLLoss2d',
           'CosineEmbeddingLoss', 'HingeEmbeddingLoss', 'MarginRankingLoss',
           'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss', 'CrossEntropyLoss',
           'TripletMarginLoss', 'PoissonNLLLoss', 'CenterLoss', 'CenterLoss2', 'FocalLoss', 'HistogramLoss',
           'MultiClassHingeLoss', 'SoftmaxLoss', 'AngleLoss', 'AllContrastiveLoss', 'AllCosineEmbeddingLoss',
           'ATriplet', 'ABatchAll', 'AHardPair', 'NCA', 'MCALoss', 'ClusterNCALoss', 'CenterNCALoss', 'SoftmaxNeigLoss',
           'NeighbourHardLoss', 'MeanBinDevianceLoss', 'MarginPositiveLoss', 'MarginDevianceLoss', 'KmeanLoss',
            'Grad_NCA', 'GaussianMetricLoss', 'GaussianLDA', 'DistanceMatchLoss',
           'DistWeightBinDevianceLoss', 'DistWeightDevBranchLoss', 'DistWeightContrastiveLoss', 'DevianceLoss',
            'BinDevianceLoss', 'BinBranchLoss', 'BatchHardLoss', 'BatchAll', 'ContrastiveNewLoss',
           'HistogramNewLoss', 'DistWeightNeighbourLoss', 'DivergenceLoss', 'DistWeightLoss', 'DecorLoss', 'JSDivLoss',
           'EnsembleDWNeighbourLoss', 'HDCLoss', 'HDC_Bin_Loss', 'BDWNeighbourLoss', 'OriBinLoss', 'NeighbourLoss',
           ]
