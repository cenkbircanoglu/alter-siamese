#!/usr/bin/env bash

# 23615

EPOCHS=1

for data in utkface_age_gender_race
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2  \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 64 --height 64 --channel 3 \
            --network alex_64 --embedding 616 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Siamese
    for loss in  ContrastiveLoss
    do
          python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 1 \
            --network siamese_alex_64 --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
    for loss in  CosineEmbeddingLoss MarginRankingLoss
    do
          python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 1 \
            --network siamese_alex_64 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Triplet
    for loss in TripletMarginLoss
    do
          python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
            --network triplet_alex_64 --embedding 128 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
done
