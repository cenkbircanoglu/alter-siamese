#!/usr/bin/env bash

for network in net_28 alex_28 dense_28
do
    for data in mnist
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