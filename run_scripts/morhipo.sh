#!/usr/bin/env bash


python __main__.py siamese --data_name morhipo  --width 64 --height 64 --channel 3 \
--network siamese_alex_64 --embedding 128 --epochs 500 --loss ContrastiveLoss --negative 1 --positive 0 \
 --loader_name pair_loaders


python __main__.py siamese --data_name morhipo  --width 224 --height 224 --channel 3 \
--network siamese_alex_224 --embedding 128 --epochs 500 --loss ContrastiveLoss --negative 1 --positive 0 \
 --loader_name pair_loaders
