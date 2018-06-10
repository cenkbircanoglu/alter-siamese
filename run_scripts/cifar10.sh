#!/usr/bin/env bash

# 50000

EPOCHS=500
for network in net_32 alex_32 dense_32
do
    for data in cifar10
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss FocalLoss CenterLoss SoftmaxLoss MultiClassHingeLoss  HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 32 --height 32 --channel 3 \
                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done
    done
done

#
#for network in angle_net_32 angle_alex_32 #angle_dense_32
#do
#    for data in cifar10
#    do
#        # Listwise
#        for loss in AngleLoss
#        do
#              python __main__.py listwise --data_name $data --width 32 --height 32 --channel 3 \
#                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
#        done
#    done
#done