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

from losses.a_batch_all import ABatchAll
from losses.a_hard_pair import AHardPair
from losses.a_triplet import ATriplet
from losses.angle import AngleLoss
from losses.batch_all import BatchAll
from losses.batch_hard import BatchHardLoss
from losses.bdw_neighbour_loss import BDWNeighbourLoss
from losses.bin_branch_loss import BinBranchLoss
from losses.bin_deviance_loss import BinDevianceLoss
#from losses.branch_knn_softmax import BranchKNNSoftmax
from losses.center import CenterLoss
from losses.center2 import CenterLoss2
from losses.center_loss import CenterLoss as CenterNewLoss
from losses.center_nca_loss import CenterNCALoss
from losses.center_triplet import CenterTripletLoss
from losses.cluster_nca_loss import ClusterNCALoss
from losses.contrastive import ContrastiveLoss  # Siamese (`1` or `0`)
from losses.contrastive import ContrastiveLoss  as AllContrastiveLoss  # Siamese (`1` or `0`)
from losses.contrastive_loss import ContrastiveLoss as ContrastiveNewLoss
from losses.cosine_embedding import CosineEmbeddingLoss  # Siamese (`1` or `-1`)
from losses.cosine_embedding import CosineEmbeddingLoss as AllCosineEmbeddingLoss  # Siamese (`1` or `-1`)
from losses.decor_loss import DecorLoss
from losses.deviance_loss import DevianceLoss
from losses.dist_weight_contrastive_loss import DistWeightContrastiveLoss
from losses.dist_weight_dev_branch_loss import DistWeightDevBranchLoss
from losses.dist_weight_deviance_loss import DistWeightBinDevianceLoss
from losses.dist_weight_loss import DistWeightLoss
from losses.dist_weight_neighbour_loss import DistWeightNeighbourLoss
from losses.distance_match_loss import DistanceMatchLoss
from losses.divergence_loss import DivergenceLoss
from losses.ensemble_dw_neighbour_loss import EnsembleDWNeighbourLoss
from losses.focal import FocalLoss
from losses.gaussian_lda import GaussianLDA
from losses.gaussian_metric import GaussianMetricLoss
from losses.grad_nca import Grad_NCA
from losses.hdc_bin_loss import HDC_Bin_Loss
from losses.hdc_loss import HDCLoss
from losses.histogram import HistogramLoss
from losses.histogram_loss import HistogramLoss as HistogramNewLoss
#from losses.js_div_knn_softmax_loss import JSDivKNNSoftmaxLoss
from losses.js_div_loss import JSDivLoss
from losses.kmean_loss import KmeanLoss
from losses.margin_deviance_loss import MarginDevianceLoss
from losses.margin_positive_loss import MarginPositiveLoss
from losses.margin_ranking import MarginRankingLoss  # Siamese (`1` or `-1`)
from losses.mca_loss import MCALoss
from losses.mean_bin_deviance_loss import MeanBinDevianceLoss
from losses.nca import NCA
from losses.softmax import SoftmaxLoss
from losses.svm import MultiClassHingeLoss
from losses.triplet import TripletMarginLoss
from neighbour_hard_loss import NeighbourHardLoss
from neighbour_loss import NeighbourLoss
from ori_bin_loss import OriBinLoss
from softmax_neig_loss import SoftmaxNeigLoss

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
