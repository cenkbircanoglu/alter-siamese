#!/usr/bin/env bash

for network in net_32 alex_32 dense_32
do
    for data in cifar100
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss HistogramLoss
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