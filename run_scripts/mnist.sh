#!/usr/bin/env bash

# 70000

EPOCHS=500
for network in net_28 alex_28 dense_28
do
    for data in mnist
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss  HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done
    done
done


for network in angle_net_28 angle_alex_28 angle_dense_28
do
    for data in mnist
    do
        # Listwise
        for loss in AngleLoss
        do
              python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done
    done
done
