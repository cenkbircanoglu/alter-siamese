#!/usr/bin/env bash


python __main__.py --config_name SiamAtt
python evaluate/svm.py --data_path results/siamatt/
python __main__.py --config_name SiameseAtt
python evaluate/svm.py --data_path results/siameseatt/

python __main__.py --config_name SiamMnist
python evaluate/svm.py --data_path results/siammnist/
python __main__.py --config_name SiameseMnist
python evaluate/svm.py --data_path results/siamesemnist/

python __main__.py --config_name SiamCifar
python evaluate/svm.py --data_path results/siamcifar/
python __main__.py --config_name SiameseCifar
python evaluate/svm.py --data_path results/siamesecifar/




