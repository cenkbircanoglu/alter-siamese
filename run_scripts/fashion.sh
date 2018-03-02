#!/usr/bin/env bash


# 29985

EPOCHS=500
for network in net_256
do
    for data in fashion
    do
        # Listwise
        for loss in CrossEntropyLoss
        do
              python __main__.py listwise --data_name $data --width 256 --height 256 --channel 3 \
                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done

    done
done

#
#
#EPOCHS=500
#for network in net_256 alex_256 #dense_256
#do
#    for data in fashion
#    do
#        # Listwise
#        for loss in CrossEntropyLoss MultiMarginLoss  SoftmaxLoss FocalLoss CenterLoss  MultiClassHingeLoss HistogramLoss CenterLoss2
#        do
#              python __main__.py listwise --data_name $data --width 256 --height 256 --channel 3 \
#                --network $network --embedding 10 --epochs $EPOCHS --loss $loss --loader_name data_loaders
#        done
#
#        # Siamese
#        for loss in  ContrastiveLoss
#        do
#              python __main__.py siamese --data_name $data  --width 256 --height 256 --channel 3 \
#                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0 \
#                 --loader_name pair_loaders
#        done
#        for loss in  CosineEmbeddingLoss
#        do
#              python __main__.py siamese --data_name $data  --width 256 --height 256 --channel 3 \
#                --network siamese_${network} --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1 \
#                 --loader_name pair_loaders
#        done
#
#        # Triplet
#        for loss in TripletMarginLoss
#        do
#              python __main__.py triplet --data_name $data  --width 256 --height 256 --channel 3 \
#                --network triplet_${network} --embedding 128 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
#        done
#    done
#done
