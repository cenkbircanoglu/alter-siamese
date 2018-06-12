#!/usr/bin/env bash

for network in net_64 alex_64 dense_64
do
    for data in cacd2000_age
    do
        for loss in  ContrastiveLoss
        do
                python  evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
        done
        for loss in  CosineEmbeddingLoss
        do
                python  evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss}
        done
        for loss in TripletMarginLoss
        do
                python  evaluate/svm.py --data_path results/${data}/triplet_${network}/${loss}
        done
    done
done