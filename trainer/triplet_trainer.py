from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from tqdm import tqdm


def run():
    from config import get_config
    config = get_config()

    print('%s/ckpt.pth.tar' % config.result_dir)
    if os.path.exists('%s/ckpt.pth.tar' % config.result_dir):
        return True
    print("Triplet Trainer Not Return")
    import models
    from utils.make_dirs import create_dirs
    from datasets import loaders
    from torchsample.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from torchsample.metrics import CategoricalAccuracy
    from torchsample.modules import ModuleTrainer
    create_dirs()
    cuda_device = -1
    tr_data_loader, val_data_loader, te_data_loader = loaders.online_triplet_loaders()

    model = getattr(models, config.network).get_network()(channel=config.network_channel,
                                                          embedding_size=config.embedding)
    from losses.online_triplet import OnlineTripletLoss
    from datasets.data_utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
        SemihardNegativeTripletSelector
    margin = 1.

    if args.selector == 'AllTripletSelector':
        criterion = OnlineTripletLoss(margin, AllTripletSelector())
    elif args.selector == 'HardestNegativeTripletSelector':
        criterion = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin))
    elif args.selector == 'RandomNegativeTripletSelector':
        criterion = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    elif args.selector == 'SemihardNegativeTripletSelector':
        criterion = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))

    if config.cuda:
        model.cuda()
        criterion.cuda()
    trainer = ModuleTrainer(model)
    epochs = config.epochs

    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(config.result_dir, save_best_only=True, verbose=1),
                 CSVLogger("%s/logger.csv" % config.result_dir)]

    metrics = []
    if config.loader_name == 'data_loaders' and 'Angle' not in config.loss:
        metrics.append(CategoricalAccuracy(top_k=1))
    trainer.compile(loss=criterion, optimizer='adam', metrics=metrics)
    trainer.set_callbacks(callbacks)
    if config.cuda:
        cuda_device = 0
    start_time = time.time()
    trainer.fit_loader(tr_data_loader, val_loader=val_data_loader, num_epoch=epochs, verbose=2,
                       cuda_device=cuda_device)
    end_time = time.time()

    with open("%s/app.log" % config.result_dir, mode="a") as f:
        f.write("%s\n" % str(model))
        f.write("%s %s\n" % (config.loss, str(end_time - start_time)))
    tr_loss = trainer.evaluate_loader(tr_data_loader, cuda_device=cuda_device)
    print(tr_loss)
    val_loss = trainer.evaluate_loader(val_data_loader, cuda_device=cuda_device)
    te_loss = trainer.evaluate_loader(te_data_loader, cuda_device=cuda_device)
    print(te_loss)
    with open(config.log_path, "a") as f:
        f.write('Train %s\nVal:%s\nTest:%s\n' % (str(tr_loss), str(val_loss), te_loss))

    tr_data_loader, val_data_loader, te_data_loader = loaders.data_loaders(train=False, val=True)

    tr_y_pred = trainer.predict_loader(tr_data_loader, cuda_device=cuda_device)
    save_embeddings(tr_y_pred, '%s/train_embeddings.csv' % config.result_dir)
    save_labels(tr_data_loader, '%s/train_labels.csv' % config.result_dir)

    val_y_pred = trainer.predict_loader(val_data_loader, cuda_device=cuda_device)
    save_embeddings(val_y_pred, '%s/val_embeddings.csv' % config.result_dir)
    save_labels(val_data_loader, '%s/val_labels.csv' % config.result_dir)

    te_y_pred = trainer.predict_loader(te_data_loader, cuda_device=cuda_device)
    save_embeddings(te_y_pred, '%s/test_embeddings.csv' % config.result_dir)
    save_labels(te_data_loader, '%s/test_labels.csv' % config.result_dir)


def save_embeddings(data, outputfile):
    with open(outputfile, 'a') as f:
        if type(data) == list:
            data = data[0]
        np.savetxt(f, data.data.cpu().numpy())


def save_labels(loader, outputfile):
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        img, label = data
        with open(outputfile, 'a') as f:
            np.savetxt(f, label.numpy())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--data_name', type=str, default="mnist")
    parser.add_argument('--network', type=str, default="net_28")
    parser.add_argument('--embedding', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--loss', type=str, default="OnlineTripletLossAllTripletSelector")
    parser.add_argument('--selector', type=str, default="AllTripletSelector")
    parser.add_argument('--epochs', type=int, default=10)

    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    set_config("siamese", **kwargs)

    run()
