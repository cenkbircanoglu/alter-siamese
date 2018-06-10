#!/usr/bin/env bash


# 24998
EPOCHS=500
for network in net_224 alex_224 dense_224
do
    for data in books
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss  HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 224 --height 224 --channel 3 \
                --network $network --embedding 32 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done

    done
done


#for network in angle_net_224 angle_alex_224 #angle_dense_224
#do
#    for data in books
#    do
#        # Listwise
#        for loss in AngleLoss
#        do
#              python __main__.py listwise --data_name $data --width 224 --height 224 --channel 3 \
#                --network $network --embedding 32 --epochs $EPOCHS --loss $loss --loader_name data_loaders
#        done
#
#    done
#done
