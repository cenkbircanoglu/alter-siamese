#!/usr/bin/env bash


# 24998
EPOCHS=500
for network in net_224 alex_224 dense_224
do
    for data in books
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 224 --height 224 --channel 3 \
                --network $network --embedding 32 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done

        # Siamese
        for loss in  ContrastiveLoss
        do
              python __main__.py siamese --data_name $data  --width 224 --height 224 --channel 3 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0 \
                 --loader_name pair_loaders
        done
        for loss in  CosineEmbeddingLoss
        do
              python __main__.py siamese --data_name $data  --width 224 --height 224 --channel 3 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1 \
                 --loader_name pair_loaders
        done

        # Triplet
        for loss in TripletMarginLoss
        do
              python __main__.py triplet --data_name $data  --width 224 --height 224 --channel 3 \
                --network triplet_${network} --embedding 128 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
        done
    done
done

for network in inception_resnet_v2_299
do
    for data in books
    do
        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss  FocalLoss SoftmaxLoss CenterLoss  MultiClassHingeLoss HistogramLoss
        do
              python __main__.py listwise --data_name $data --width 299 --height 299 --channel 3 \
                --network $network --embedding 32 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done
        # Siamese
        for loss in  ContrastiveLoss
        do
              python __main__.py siamese --data_name $data  --width 299 --height 299 --channel 3 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0 \
                 --loader_name pair_loaders
        done
        for loss in  CosineEmbeddingLoss
        do
              python __main__.py siamese --data_name $data  --width 299 --height 299 --channel 3 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1 \
                 --loader_name pair_loaders
        done
        # Triplet
        for loss in TripletMarginLoss
        do
              python __main__.py triplet --data_name $data  --width 299 --height 299 --channel 3 \
                --network triplet_${network} --embedding 128 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
        done
    done
done