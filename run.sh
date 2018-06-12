#!/usr/bin/env bash

sh run_scripts/mnist.sh
sh run_scripts/cifar100.sh
sh run_scripts/cifar10.sh
sh run_scripts/gamo.sh
sh run_scripts/cats_dogs.sh
sh run_scripts/cifar10_percent.sh
sh run_scripts/aloi_red2_ill.sh
sh run_scripts/cacd2000_age.sh
sh run_scripts/utkface_age.sh
sh run_scripts/fashion.sh
sh run_scripts/books.sh
sh run_scripts/marvel.sh
#sh run_scripts/fashion_percent.sh
#sh run_scripts/imagenet.sh


sh run_scripts/imsize_28.sh
sh run_scripts/imsize_32.sh
sh run_scripts/imsize_64.sh
sh run_scripts/imsize_224.sh


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


