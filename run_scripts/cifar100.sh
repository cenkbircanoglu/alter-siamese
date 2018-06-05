#!/usr/bin/env bash

# 50000

EPOCHS=500
for network in net_32 alex_32 dense_32
do
    for data in cifar100
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss SoftmaxLoss FocalLoss CenterLoss  MultiClassHingeLoss  HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 32 --height 32 --channel 3 \
                --network $network --embedding 100 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done
    done
done