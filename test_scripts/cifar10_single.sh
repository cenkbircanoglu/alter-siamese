#!/usr/bin/env bash

for network in net_32 alex_32 dense_32
do
    for data in cifar10 cifar10_10 cifar10_20 cifar10_30 cifar10_40 cifar10_50 cifar10_60
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss HistogramLoss
        do
               timeout 1200s python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done
    done
done