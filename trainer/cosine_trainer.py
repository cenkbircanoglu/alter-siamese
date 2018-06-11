from __future__ import division
from __future__ import print_function

import time

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from datasets.dataset import NetworkDataset
from losses import CosineEmbeddingLoss2


def run():
    from models.s28.net import Net
    from torchsample.metrics import CategoricalAccuracy
    from torchsample.modules import ModuleTrainer

    transform = transforms.Compose(
        [transforms.Scale((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(
            root='/Users/cenk.bircanoglu/personal/alter_siamese/data/mnist/train'),
        transform=transform,
        should_invert=False,
        channel=1,
        train=True,
        val=False
    )
    tr_data_loader = DataLoader(tr_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=128)

    val_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root='/Users/cenk.bircanoglu/personal/alter_siamese/data/mnist/val'),
        transform=transform,
        should_invert=False,
        channel=1,
        train=True,
        val=False
    )

    val_data_loader = DataLoader(val_dataset,
                                 shuffle=True,
                                 num_workers=8,
                                 batch_size=128)

    te_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root='/Users/cenk.bircanoglu/personal/alter_siamese/data/mnist/test'),
        transform=transform,
        should_invert=False,
        channel=1,
        train=True,
        val=False
    )
    te_data_loader = DataLoader(te_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=128)

    cuda_device = -1
    model = Net(channel=1, embedding_size=128)
    criterion = CosineEmbeddingLoss2()
    trainer = ModuleTrainer(model)
    metrics = []
    metrics.append(CategoricalAccuracy(top_k=1))
    trainer.compile(loss=criterion, optimizer='adam', metrics=metrics)

    start_time = time.time()
    trainer.fit_loader(tr_data_loader, val_loader=val_data_loader, num_epoch=10, verbose=2,
                       cuda_device=cuda_device)
    end_time = time.time()
    tr_loss = trainer.evaluate_loader(tr_data_loader, cuda_device=cuda_device)
    print(tr_loss)
    val_loss = trainer.evaluate_loader(val_data_loader, cuda_device=cuda_device)
    te_loss = trainer.evaluate_loader(te_data_loader, cuda_device=cuda_device)
    print(te_loss)

    tr_y_pred = trainer.predict_loader(tr_data_loader, cuda_device=cuda_device)
    save_embeddings(tr_y_pred, 'cosine/train_embeddings.csv')
    save_labels(tr_data_loader, 'cosine/train_labels.csv')

    val_y_pred = trainer.predict_loader(val_data_loader, cuda_device=cuda_device)
    save_embeddings(val_y_pred, 'cosine/val_embeddings.csv')
    save_labels(val_data_loader, 'cosine/val_labels.csv')

    te_y_pred = trainer.predict_loader(te_data_loader, cuda_device=cuda_device)
    save_embeddings(te_y_pred, 'cosine/test_embeddings.csv')
    save_labels(te_data_loader, 'cosine/test_labels.csv')


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
    run()
