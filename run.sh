#!/usr/bin/env bash


EPOCHS=1
for data in cats_and_dogs
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2  \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 64 --height 64 --channel 3 \
            --network net_64 --embedding 2 --epochs $EPOCHS --loss $loss
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

for data in aloi_red2_ill
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2  \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 64 --height 64 --channel 3 \
            --network net_64 --embedding 1000 --epochs $EPOCHS --loss $loss
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


for data in mnist
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2  \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
            --network net_28 --embedding 10 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Siamese
    for loss in  ContrastiveLoss
    do
          python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
            --network siamese_net_28 --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
    for loss in  CosineEmbeddingLoss MarginRankingLoss
    do
          python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
            --network siamese_net_28 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Triplet
    for loss in TripletMarginLoss
    do
          python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
            --network triplet_net_28 --embedding 128 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
done


for data in cifar10
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2 \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 32 --height 32 --channel 3 \
            --network net_32 --embedding 10 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Siamese
    for loss in  ContrastiveLoss
    do
          python __main__.py siamese --data_name $data  --width 32 --height 32 --channel 3 \
            --network siamese_net_32 --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
    for loss in  CosineEmbeddingLoss MarginRankingLoss
    do
          python __main__.py siamese --data_name $data  --width 32 --height 32 --channel 3 \
            --network siamese_net_32 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Triplet
    for loss in TripletMarginLoss
    do
          python __main__.py triplet --data_name $data  --width 32 --height 32 --channel 3 \
            --network triplet_net_32 --embedding 128 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
done



for data in cifar100
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss CenterLoss CenterLoss2 \
        MultiClassHingeLoss #HistogramLoss
    do
          python __main__.py listwise --data_name $data --width 32 --height 32 --channel 3 \
            --network net_32 --embedding 100 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Siamese
    for loss in  ContrastiveLoss
    do
          python __main__.py siamese --data_name $data  --width 32 --height 32 --channel 3 \
            --network siamese_net_32 --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
    for loss in  CosineEmbeddingLoss MarginRankingLoss
    do
          python __main__.py siamese --data_name $data  --width 32 --height 32 --channel 3 \
            --network siamese_net_32 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Triplet
    for loss in TripletMarginLoss
    do
          python __main__.py triplet --data_name $data  --width 32 --height 32 --channel 3 \
            --network triplet_net_32 --embedding 128 --epochs $EPOCHS --loss $loss
          python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
done



#python __main__.py listwise --data_name att  --width 100 --height 100 --channel 1 --network vgg_100 --embedding 40  --epochs $EPOCHS --loss NLLLoss
#python evaluate/svm.py --data_path results/listwise_att/ &
#python __main__.py siam --data_name att  --width 100 --height 100 --channel 1 --network siam_vgg_100 --embedding 256 --epochs $EPOCHS --loss ContrastiveLoss
#python evaluate/svm.py --data_path results/siam_att/ &
#python __main__.py siamese --data_name att  --width 100 --height 100 --channel 1 --network siamese_vgg_100 --embedding 128 --epochs $EPOCHS --loss ContrastiveLoss
#python evaluate/svm.py --data_path results/siamese_att/ &
#python __main__.py trip --data_name att  --width 100 --height 100 --channel 1 --network trip_vgg_100 --embedding 384 --epochs $EPOCHS --loss TripletMarginLoss
#python evaluate/svm.py --data_path results/trip_att/ &
#python __main__.py triplet --data_name att  --width 100 --height 100 --channel 1 --network triplet_vgg_100 --embedding 128 --epochs $EPOCHS --loss TripletMarginLoss
#python evaluate/svm.py --data_path results/triplet_att/ &
#
#
#python __main__.py listwise --data_name cifar  --width 32 --height 32 --channel 3 --network vgg_32 --embedding 10 --epochs $EPOCHS --loss NLLLoss
#python evaluate/svm.py --data_path results/listwise_cifar/ &
#python __main__.py siam --data_name cifar  --width 32 --height 32 --channel 3 --network siam_vgg_32 --embedding 256 --epochs $EPOCHS --loss ContrastiveLoss
#python evaluate/svm.py --data_path results/siam_cifar/ &
#python __main__.py siamese --data_name cifar  --width 32 --height 32 --channel 3 --network siamese_vgg_32 --embedding 128 --epochs $EPOCHS --loss ContrastiveLoss
#python evaluate/svm.py --data_path results/siamese_cifar/ &
#python __main__.py trip --data_name cifar  --width 32 --height 32 --channel 3 --network trip_vgg_32 --embedding 384 --epochs $EPOCHS --loss TripletMarginLoss
#python evaluate/svm.py --data_path results/trip_cifar/ &
#python __main__.py triplet --data_name cifar  --width 32 --height 32 --channel 3 --network triplet_vgg_32 --embedding 128 --epochs $EPOCHS --loss TripletMarginLoss
#python evaluate/svm.py --data_path results/triplet_cifar/ &


