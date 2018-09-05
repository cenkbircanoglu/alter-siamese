#!/usr/bin/env bash


train ()
{
    for network in net_${img_dim} alex_${img_dim} dense_${img_dim}  # mynet_$[img_dim}
    do

        # Listwise
        for loss in CrossEntropyLoss MultiMarginLoss FocalLoss CenterLoss SoftmaxLoss MultiClassHingeLoss  HistogramLoss
        do
              python __main__.py listwise --data_name $data --width $img_dim --height $img_dim --channel $channel \
                --network $network --embedding $embedding --epochs $EPOCHS --loss $loss --loader_name data_loaders
        done

#        # Siamese
#        for loss in  ContrastiveLoss
#        do
#              python __main__.py siamese --data_name $data  --width $img_dim --height $img_dim --channel $channel \
#                --network siamese_${network} --embedding $embedding_size --epochs $EPOCHS --loss $loss --negative 1 --positive 0 \
#                 --loader_name pair_loaders
#        done
#
#        # Siamese
#        for loss in  CosineEmbeddingLoss
#        do
#              python __main__.py siamese --data_name $data  --width $img_dim --height $img_dim --channel $channel \
#                --network siamese_${network} --embedding $embedding_size --epochs $EPOCHS --loss $loss --negative -1 --positive 1 \
#                 --loader_name pair_loaders
#        done
#
#        # Triplet
#        for loss in TripletMarginLoss
#        do
#              python __main__.py triplet --data_name $data  --width $img_dim --height $img_dim --channel $channel \
#                --network triplet_${network} --embedding $embedding_size --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
#        done


        for selector in HardNegativePairSelector AllPositivePairSelector
        do
            python trainer/contrastive_trainer.py --data_name $data --width $img_dim --height $img_dim --channel $channel \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineContrastiveLoss${selector} \
                 --selector $selector

            python trainer/contrastive_timer.py --data_name $data --width $img_dim --height $img_dim --channel $channel \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineContrastiveLoss${selector} \
                 --selector $selector
        done

        for selector in AllTripletSelector HardestNegativeTripletSelector RandomNegativeTripletSelector SemihardNegativeTripletSelector
        do
           python trainer/triplet_trainer.py --data_name $data --width $img_dim --height $img_dim --channel $channel \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineTripletLoss${selector} \
                 --selector $selector

            python trainer/triplet_timer.py --data_name $data --width $img_dim --height $img_dim --channel $channel \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineTripletLoss${selector} \
                 --selector $selector
        done

        for selector in HardNegativePairSelector AllPositivePairSelector
        do
           python trainer/cosine_trainer.py --data_name $data --width $img_dim --height $img_dim --channel $channel \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineCosineLoss${selector} \
                 --selector $selector
           python trainer/cosine_timer.py --data_name $data --width $img_dim --height $img_dim --channel $channel \
                --network $network --embedding 128 --epochs $EPOCHS --loss OnlineCosineLoss${selector} \
                 --selector $selector
        done
    done

}

EPOCHS=500
embedding_size=128

img_dim=28
channel=1
embedding=10
data=mnist
train

channel=3

img_dim=32

embedding=10

for data in cifar10 #cifar10_10 cifar10_20 cifar10_30 cifar10_40 cifar10_50 cifar10_60
do
    train
done

embedding=100
data=cifar100
train

img_dim=64

embedding=1000
data=aloi_red2_ill
train

embedding=49
data=cacd2000_age
train

embedding=2
data=cats_dogs
train

embedding=7
data=gamo
train

embedding=98
data=utkface_age
train

embedding=6
data=trasnet
train

#embedding=1000
#data=imagenet
#train

img_dim=224

embedding=196
data=cars_196
train

embedding=200
data=cub_200_2011
train

embedding=10
for data in fashion #fashion_10 fashion_20 fashion_30 fashion_40 fashion_50 fashion_60
do
    train
done

embedding=86
data=marvel
train

embedding=32
data=books
train

#embedding=22634
#data=products
#train
#
