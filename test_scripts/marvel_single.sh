#!/usr/bin/env bash

for network in net_224 alex_224 dense_224
do
    for data in marvel
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss HistogramLoss
        do
               timeout 1200s python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done
    done
done