from __future__ import division
from __future__ import print_function
import time

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torchvision.transforms as transforms

from datasets.dataset import NetworkDataset
from losses.triplet_2 import module_hook

def run():
    from losses import TripletMarginLoss2
    from models.s28.net import Net
    from torchsample.metrics import CategoricalAccuracy
    from torchsample.modules import ModuleTrainer

    transform = transforms.Compose(
        [transforms.Scale((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_dataset = NetworkDataset(
        image_folder_dataset=datasets.ImageFolder(root='/media/cenk/2TB1/alter_siamese/data/mnist/train'),
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
        image_folder_dataset=datasets.ImageFolder(root='/media/cenk/2TB1/alter_siamese/data/mnist/val'),
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
        image_folder_dataset=datasets.ImageFolder(root='/media/cenk/2TB1/alter_siamese/data/mnist/test'),
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
    criterion = TripletMarginLoss2()
    criterion.register_backward_hook(module_hook)
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

    val_y_pred = trainer.predict_loader(val_data_loader, cuda_device=cuda_device)


    te_y_pred = trainer.predict_loader(te_data_loader, cuda_device=cuda_device)



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
