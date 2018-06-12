#!/usr/bin/env bash

# 70000

EPOCHS=500
for network in net_28 alex_28 #dense_28
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


for network in net_28 alex_28 #dense_28
do
    for data in mnist
    do
        for selector in HardNegativePairSelector AllPositivePairSelector
        do
            python trainer/contrastive_trainer.py --data_name $data --width 28 --height 28 --channel 1 \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineContrastiveLoss${selector} \
                 --selector $selector
        done

        for selector in AllTripletSelector HardestNegativeTripletSelector RandomNegativeTripletSelector SemihardNegativeTripletSelector
        do
            python trainer/triplet_trainer.py --data_name $data --width 28 --height 28 --channel 1 \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineTripletLoss${selector} \
                 --selector $selector
        done

        for selector in HardNegativePairSelector AllPositivePairSelector
        do
            python trainer/cosine_trainer.py --data_name $data --width 28 --height 28 --channel 1 \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineCosineLoss${selector} \
                 --selector $selector
        done
    done
done