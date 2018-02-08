from __future__ import division
from __future__ import print_function

import numpy as np
from torchsample.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from torchsample.metrics import CategoricalAccuracy
from torchsample.modules import ModuleTrainer
from torchvision.models import DenseNet
from tqdm import tqdm

from datasets.loaders import data_loaders


def run():
    from config import get_config
    config = get_config()
    import losses
    import models
    from utils.make_dirs import create_dirs
    from datasets import loaders
    create_dirs()

    tr_data_loader, te_data_loader = getattr(loaders, config.loader_name)()

    model = getattr(models, config.network).get_network()(channel=config.network_channel,
                                                          embedding_size=config.embedding)
    #model = DenseNet(num_classes=10)
    model.cuda()
    trainer = ModuleTrainer(model)

    callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint(config.result_dir),
                 CSVLogger("%s/logger.csv" % config.result_dir)]
    metrics = []
    if config.loader_name == 'data_loaders':
        metrics.append(CategoricalAccuracy(top_k=1))
    trainer.compile(loss=getattr(losses, config.loss)(), optimizer='adam', metrics=metrics)
    trainer.set_callbacks(callbacks)

    trainer.fit_loader(tr_data_loader, val_loader=te_data_loader, num_epoch=config.epochs, verbose=2, cuda_device=0)

    tr_loss = trainer.evaluate_loader(tr_data_loader, cuda_device=0)
    print(tr_loss)
    te_loss = trainer.evaluate_loader(te_data_loader, cuda_device=0)
    print(te_loss)

    tr_data_loader, te_data_loader = data_loaders()
    tr_y_pred = trainer.predict_loader(tr_data_loader, cuda_device=0)
    te_y_pred = trainer.predict_loader(te_data_loader, cuda_device=0)
    with open(config.log_path, "a") as f:
        f.write('Train: %s\nTest: %s\n' % (str(tr_loss), te_loss))

    with open('%s/train_embeddings.csv' % config.result_dir, 'a') as f:
        np.savetxt(f, tr_y_pred.data.numpy())
    with open('%s/test_embeddings.csv' % config.result_dir, 'a') as f:
        np.savetxt(f, te_y_pred.data.numpy())
    save_labels(tr_data_loader, '%s/train_labels.csv' % config.result_dir)
    save_labels(te_data_loader, '%s/test_labels.csv' % config.result_dir)


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
    parser.add_argument('--loss', type=str, default="NLLLoss")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--negative', type=int, default=0)
    parser.add_argument('--positive', type=int, default=1)

    from config import set_config, get_config

    args = parser.parse_args()

    kwargs = vars(args)
    set_config("siamese", **kwargs)

    run()
