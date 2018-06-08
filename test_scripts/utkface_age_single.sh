#!/usr/bin/env bash

for network in net_64 alex_64 dense_64
do
    for data in utkface_age
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss HistogramLoss
        do
               timeout 1200s python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done
    done
done