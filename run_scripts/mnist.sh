#!/usr/bin/env bash

# 70000

EPOCHS=300
for network in net_28 alex_28 dense_28
do
    for data in mnist
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss
        do
              python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
              python evaluate/svm.py --data_path results/${data}/${network}/${loss} &
        done

        # Siamese
        for loss in  ContrastiveLoss
        do
              python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0 \
                 --loader_name pair_loaders
              python evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss} &
        done
        for loss in  CosineEmbeddingLoss MarginRankingLoss
        do
              python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1 \
                 --loader_name pair_loaders
              python evaluate/svm.py --data_path results/${data}/siamese_${network}/${loss} &
        done

        # Triplet
        for loss in TripletMarginLoss
        do
              python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
                --network triplet_${network} --embedding 128 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
              python evaluate/svm.py --data_path results/${data}/triplet_${network}/${loss} &
        done

        # Histogram
        for loss in HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name histogram_loaders
              python evaluate/svm.py --data_path results/${data}/${network}/${loss} &
        done
    done
done


