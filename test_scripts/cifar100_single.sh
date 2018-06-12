#!/usr/bin/env bash

for network in net_32 alex_32 dense_32
do
    for data in cifar100
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss HistogramLoss
        do
                python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done
    done
done