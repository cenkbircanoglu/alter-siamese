#!/usr/bin/env bash


EPOCHS=10
for data in mnist
do
    # Listwise
    for loss in CrossEntropyLoss MultiMarginLoss NLLLoss FocalLoss SoftmaxLoss
    do
        echo python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
            --network net_28 --embedding 10 --epochs $EPOCHS --loss $loss
        echo python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Siamese
    for loss in  ContrastiveLoss
    do
        echo python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
            --network siamese_net_28 --embedding 128 --epochs $EPOCHS --loss $loss --negative 1 --positive 0
        echo python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
    for loss in  CosineEmbeddingLoss
    do
        echo python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
            --network siamese_net_28 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1 --positive 1
        echo python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done

    # Triplet
    for loss in TripletMarginLoss
    do
        echo python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
            --network triplet_net_28 --embedding 128 --epochs $EPOCHS --loss $loss
        echo python evaluate/svm.py --data_path results/${data}/${loss}/ &
    done
done


#for data in mnist
#do
#    # Listwise
#    for loss in CrossEntropyLoss  MultiLabelMarginLoss MultiLabelSoftMarginLoss \
#     MultiMarginLoss NLLLoss PoissonNLLLoss SoftMarginLoss CenterLoss CenterLoss2 FocalLoss HistogramLoss \
#           LSoftmaxLinear soft_nll_loss multiClassHingeLoss tsne_loss NEG_loss MVNLoss MGLoss SoftmaxLoss
#    do
#        python __main__.py listwise --data_name $data --width 28 --height 28 --channel 1 \
#            --network net_28 --embedding 10 --epochs $EPOCHS --loss $loss
#        python evaluate/svm.py --data_path results/${loss}_${data}/ &
#    done
#
#    # Siamese
#    for loss in  ContrastiveLoss
#    do
#        #python __main__.py siam --data_name $data  --width 28 --height 28 --channel 1 \
#        #    --network siam_net_28 --embedding 256 --epochs $EPOCHS --loss $loss
#        #python evaluate/svm.py --data_path results/${loss}_${data}/ &
#        python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
#            --network siamese_net_28 --embedding 128 --epochs $EPOCHS --loss $loss
#        python evaluate/svm.py --data_path results/${loss}_${data}/ &
#    done
#    for loss in  MarginRankingLoss CosineEmbeddingLoss
#    do
#        #python __main__.py siam --data_name $data  --width 28 --height 28 --channel 1 \
#        #    --network siam_net_28 --embedding 256 --epochs $EPOCHS --loss $loss --negative -1
#        #python evaluate/svm.py --data_path results/${loss}_${data}/ &
#        python __main__.py siamese --data_name $data  --width 28 --height 28 --channel 1 \
#            --network siamese_net_28 --embedding 128 --epochs $EPOCHS --loss $loss --negative -1
#        python evaluate/svm.py --data_path results/${loss}_${data}/ &
#    done
#
#    # Triplet
#    for loss in TripletMarginLoss
#    do
#        #python __main__.py trip --data_name $data  --width 28 --height 28 --channel 1 \
#        #    --network trip_net_28 --embedding 384 --epochs $EPOCHS --loss $loss
#        #python evaluate/svm.py --data_path results/${loss}_${data}/ &
#        python __main__.py triplet --data_name $data  --width 28 --height 28 --channel 1 \
#            --network triplet_net_28 --embedding 128 --epochs $EPOCHS --loss $loss
#        python evaluate/svm.py --data_path results/${loss}_${data}/ &
#    done
#done
#


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


