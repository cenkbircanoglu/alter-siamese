#!/usr/bin/env bash

# 70000
################ OK ######################
# python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss CenterNCALoss --loader_name data_loaders
# python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss NeighbourHardLoss --loader_name data_loaders
# python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss GaussianMetricLoss --loader_name data_loaders
################ OK ######################

python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss ATriplet --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss ABatchAll --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss AHardPair --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss NCA --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss MCALoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss ClusterNCALoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss SoftmaxNeigLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss MeanBinDevianceLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss MarginPositiveLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss MarginDevianceLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss KmeanLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss JSDivKNNSoftmaxLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss Grad_NCA --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss GaussianLDA --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DistanceMatchLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DistWeightBinDevianceLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DistWeightDevBranchLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DistWeightContrastiveLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DevianceLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss BranchKNNSoftmax --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss BinDevianceLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss BinBranchLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss BatchHardLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss BatchAll --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss ContrastiveNewLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss HistogramNewLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DistWeightNeighbourLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DivergenceLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DistWeightLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss DecorLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss JSDivLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss EnsembleDWNeighbourLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss HDCLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss HDC_Bin_Loss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss BDWNeighbourLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss OriBinLoss --loader_name data_loaders
python __main__.py listwise --data_name mnist --width 28 --height 28 --channel 1 --network net_28 --embedding 10 --epochs 1 --loss NeighbourLoss --loader_name data_loaders


#EPOCHS=1
#for network in net_28
#do
#    for data in mnist
#    do
#        for loss in ATriplet ABatchAll AHardPair NCA MCALoss ClusterNCALoss CenterNCALoss SoftmaxNeigLoss \
#           NeighbourHardLoss MeanBinDevianceLoss MarginPositiveLoss MarginDevianceLoss KmeanLoss \
#           JSDivKNNSoftmaxLoss Grad_NCA GaussianMetricLoss GaussianLDA DistanceMatchLoss \
#           DistWeightBinDevianceLoss DistWeightDevBranchLoss DistWeightContrastiveLoss DevianceLoss \
#           BranchKNNSoftmax BinDevianceLoss BinBranchLoss BatchHardLoss BatchAll ContrastiveNewLoss \
#           HistogramNewLoss DistWeightNeighbourLoss DivergenceLoss DistWeightLoss DecorLoss JSDivLoss \
#           EnsembleDWNeighbourLoss HDCLoss HDC_Bin_Loss BDWNeighbourLoss OriBinLoss NeighbourLoss
#        do
#              python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
#                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
#        done
#    done
#done
