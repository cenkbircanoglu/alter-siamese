#!/usr/bin/env bash

for network in net_224 alex_224 dense_224
do
    for data in books
    do
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss CenterLoss2  MultiClassHingeLoss HistogramLoss
        do
                python  evaluate/svm.py --data_path results/${data}/${network}/${loss}
        done
    done
done