#!/usr/bin/env bash

for network in net_64 alex_64 dense_64
do
    for data in gamo
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss HistogramLoss
        do
                python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done
    done
done