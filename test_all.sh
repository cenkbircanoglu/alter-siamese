#!/usr/bin/env bash


test ()
{

    for network in net_${img_dim} alex_${img_dim} dense_${img_dim} mynet_${img_dim}
    do
#        for loss in  ContrastiveLoss
#        do
#               python  evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
#        done
#
#        for loss in  CosineEmbeddingLoss
#        do
#               python  evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
#        done
#
#        for loss in TripletMarginLoss
#        do
#                python  evaluate/svm.py --data_path results/${data}/triplet_${network}/${loss}
#        done

        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss #HistogramLoss
        do
               python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done

        for loss in OnlineContrastiveLossAllPositivePairSelector OnlineContrastiveLossHardNegativePairSelector \
         OnlineCosineLossAllPositivePairSelector OnlineCosineLossHardNegativePairSelector OnlineTripletLossAllTripletSelector \
            OnlineTripletLossHardestNegativeTripletSelector OnlineTripletLossRandomNegativeTripletSelector  \
            OnlineTripletLossSemihardNegativeTripletSelector
        do
               timeout 600 python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done

    done
}


img_dim=28
data=mnist
test

img_dim=32

for data in cifar10 #cifar10_10 cifar10_20 cifar10_30 cifar10_40 cifar10_50 cifar10_60 cifar100
do
    test
done


img_dim=64

for data in aloi_red2_ill cacd2000_age cats_dogs gamo utkface_age #imagenet
do
    test
done

#img_dim=224
#
#for data in books car_196 cub_200_2011 fashion fashion_10 fashion_20 fashion_30 fashion_40 fashion_50 fashion_60 \
#    marvel products
#do
#    test
#done
#
#python  utils/loss_graph.py
#
#python  utils/visualize_svm_results.py