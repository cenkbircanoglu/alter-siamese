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



python __main__.py --config_name TripAtt
python evaluate/svm.py --data_path results/tripatt/
python __main__.py --config_name TripletAtt
python evaluate/svm.py --data_path results/tripletatt/

python __main__.py --config_name TripMnist
python evaluate/svm.py --data_path results/tripmnist/
python __main__.py --config_name TripletMnist
python evaluate/svm.py --data_path results/tripletmnist/

python __main__.py --config_name TripCifar
python evaluate/svm.py --data_path results/tripcifar/
python __main__.py --config_name TripletCifar
python evaluate/svm.py --data_path results/tripletcifar/
