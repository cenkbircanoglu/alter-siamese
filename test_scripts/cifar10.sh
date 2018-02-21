#!/usr/bin/env bash

for network in net_32 alex_32 dense_32
do
    for data in cifar10 cifar10_10 cifar10_20 cifar10_30 cifar10_40 cifar10_50 cifar10_60
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss HistogramLoss
        do
              python evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done
        for loss in  ContrastiveLoss
        do
              python evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
        done
        for loss in  CosineEmbeddingLoss
        do
              python evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
        done
        for loss in TripletMarginLoss
        do
              python evaluate/svm.py --data_path results/${data}/triplet_${network}/${loss}
        done
    done
done