#!/usr/bin/env bash


# 29985

EPOCHS=500
for network in net_224 alex_224 dense_224
do
    for data in fashion_10 fashion_20 fashion_30 fashion_40 fashion_50 fashion_60
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss  HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 224 --height 224 --channel 3 \
                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done
    done
done


#for network in angle_net_224 angle_alex_224 #angle_dense_224
#do
#    for data in fashion_10 fashion_20 fashion_30 fashion_40 fashion_50 fashion_60
#    do
#        # Listwise
#        for loss in AngleLoss
#        do
#              python __main__.py listwise --data_name $data --width 224 --height 224 --channel 3 \
#                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
#        done
#    done
#done
