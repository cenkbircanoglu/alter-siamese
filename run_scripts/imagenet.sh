#!/usr/bin/env bash

# 1331167

EPOCHS=500
for network in net_64 alex_64 dense_64
do
    for data in imagenet
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss
        do
              python __main__.py listwise --data_name $data --width 64 --height 64 --channel 3 \
                --network $network --embedding 1000 --epochs $EPOCHS --loss $loss --loader_name data_loaders
              python evaluate/svm.py --data_path results/${data}/${network}/${loss} &
        done

        # Siamese
        for loss in  ContrastiveLoss
        do
              python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 3 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0 \
                 --loader_name pair_loaders
              python evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss} &
        done
        for loss in  CosineEmbeddingLoss MarginRankingLoss
        do
              python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 3 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1 \
                 --loader_name pair_loaders
              python evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss} &
        done

        # Triplet
        for loss in TripletMarginLoss
        do
              python __main__.py triplet --data_name $data  --width 64 --height 64 --channel 3 \
                --network triplet_${network} --embedding 128 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
              python evaluate/svm.py --data_path results/${data}/triplet_${network}/${loss} &
        done

        # Histogram
        for loss in HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 64 --height 64 --channel 3 \
                --network $network --embedding 1000 --epochs $EPOCHS --loss $loss --loader_name data_loaders
              python evaluate/svm.py --data_path results/${data}/${network}/${loss} &
        done
    done
done
