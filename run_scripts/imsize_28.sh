#!/usr/bin/env bash

# 70000

EPOCHS=500
for network in net_28 alex_28 dense_28
do
    for data in mnist
    do
        # Siamese
        for loss in  ContrastiveLoss
        do
              python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0 \
                 --loader_name pair_loaders
        done
        for loss in  CosineEmbeddingLoss
        do
              python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1 \
                 --loader_name pair_loaders
        done

        # Triplet
        for loss in TripletMarginLoss
        do
              python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
                --network triplet_${network} --embedding 128 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
        done
    done
done

for network in mynet_28
do
    for data in mnist
    do

        # Triplet
        for loss in TripletMarginLoss
        do
              python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
                --network triplet_${network} --embedding 256 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
        done

         # Triplet
        for loss in CosineEmbeddingLoss
        do
              python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
                --network siamese_${network} --embedding 256 --epochs $EPOCHS --loss $loss  --negative -1 --positive 1 \
                 --loader_name pair_loaders
        done

         # Triplet
        for loss in  ContrastiveLoss
        do
              python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
                --network siamese_${network} --embedding 256 --epochs $EPOCHS --loss $loss  --negative 1 --positive 0 \
                 --loader_name pair_loaders
        done
    done
done


