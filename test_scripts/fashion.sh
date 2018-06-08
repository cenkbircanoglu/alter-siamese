#!/usr/bin/env bash

for network in net_224 alex_224 dense_224
do
    for data in fashion fashion_10 fashion_20 fashion_30 fashion_40 fashion_50 fashion_60
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