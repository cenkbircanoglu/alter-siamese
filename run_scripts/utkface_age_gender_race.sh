#!/usr/bin/env bash

EPOCHS=1
# TODO train ve test label sayıları tutmuyor

for data in utkface_age_gender_race
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2  \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 64 --height 64 --channel 3 \
            --network net_64 --embedding 733 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Siamese
    for loss in  ContrastiveLoss
    do
          python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 1 \
            --network siamese_net_64 --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
    for loss in  CosineEmbeddingLoss MarginRankingLoss
    do
          python __main__.py siamese --data_name $data  --width 64 --height 64 --channel 1 \
            --network siamese_net_64 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Triplet
    for loss in TripletMarginLoss
    do
          python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
            --network triplet_net_64 --embedding 128 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
done
