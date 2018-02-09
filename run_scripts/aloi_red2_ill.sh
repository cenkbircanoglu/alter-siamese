#!/usr/bin/env bash

# 24000
EPOCHS=20

for data in aloi_red2_ill
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2  \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 64 --height 64 --channel 3 \
            --network alex_64 --embedding 1000 --epochs $EPOCHS --loss $loss --loader_name data_loaders
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Siamese
    for loss in  ContrastiveLoss
    do
          python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 1 \
            --network siamese_alex_64 --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0  --loader_name pair_loaders
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
    for loss in  CosineEmbeddingLoss MarginRankingLoss
    do
          python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 1 \
            --network siamese_alex_64 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1  --loader_name pair_loaders
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Triplet
    for loss in TripletMarginLoss
    do
          python __main__.py triplet --data_name $data  --width 64 --height 64 --channel 1 \
            --network triplet_alex_64 --embedding 128 --epochs $EPOCHS --loss $loss  --loader_name triplet_loaders
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
done
