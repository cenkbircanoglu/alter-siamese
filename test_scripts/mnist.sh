#!/usr/bin/env bash

for network in net_28 alex_28 dense_28 mynet_28
do
    for data in mnist
    do
        for loss in  ContrastiveLoss
        do
               timeout 1200s python  evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
        done
        for loss in  CosineEmbeddingLoss
        do
               timeout 1200s python  evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
        done
        for loss in TripletMarginLoss
        do
               timeout 1200s python  evaluate/svm.py --data_path results/${data}/triplet_${network}/${loss}
        done
    done
done