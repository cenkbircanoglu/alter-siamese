import datetime
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


if __name__ == '__main__':
    fontsize = 7.5

    names = glob.glob("../results/**/**/**/logger.csv")
    for name in names:

        try:
            data = pd.read_csv(name)
        except Exception as e:
            print(name)
            os.remove(name)
            continue
        print(len(data))
        data.plot(x='epoch', y=["loss", "val_loss"], figsize=(9, 6), title=name.split("/")[-2])
        plt.legend(["train", "validation"], fontsize=fontsize)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.savefig(name.replace("logger.csv", "loss.jpg"), dpi=300)
        plt.close()

    # for dataname in glob.glob("../results/**/"):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     names = glob.glob(os.path.join(dataname, "**/**/logger.csv"))
    #     for name in names:
    #         print(name)
    #         data = pd.read_csv(name)
    #         ax.plot(data['epoch'], data["loss"], label='%s_train' % name.split("/")[-2])
    #         ax.plot(data['epoch'], data["val_loss"], label='%s_val' % name.split("/")[-2])
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=fontsize)
    #     plt.tight_layout(rect=[0, 0, 0.75, 1])
    #     plt.savefig(os.path.join(dataname, "losses.jpg"), dpi=300)
    #     plt.close()
    #
    # for dataname in glob.glob("../results/**/"):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     names = glob.glob(os.path.join(dataname, "**/**/logger.csv"))
    #     for name in names:
    #         print(name)
    #         data = pd.read_csv(name)
    #         ax.plot(data['epoch'], data["loss"], label='%s' % name.split("/")[-2])
    #     plt.ylabel('Train Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=fontsize)
    #     plt.tight_layout(rect=[0, 0, 0.75, 1])
    #     plt.savefig(os.path.join(dataname, "train_losses.jpg"), dpi=300)
    #     plt.close()
    #
    # for dataname in glob.glob("../results/**/"):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     names = glob.glob(os.path.join(dataname, "**/**/logger.csv"))
    #     for name in names:
    #         print(name)
    #         data = pd.read_csv(name)
    #         ax.plot(data['epoch'], data["val_loss"], label='%s' % name.split("/")[-2])
    #     plt.ylabel('Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=fontsize)
    #     plt.tight_layout(rect=[0, 0, 0.75, 1])
    #     plt.savefig(os.path.join(dataname, "val_losses.jpg"), dpi=300)
    #     plt.close()
    #
    # for net in ["alex", "dense", "net"]:
    #     for dataname in glob.glob("../results/**/"):
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         names = glob.glob(os.path.join(dataname, "%s*/**/logger.csv" % net))
    #         for name in names:
    #             print(name)
    #             data = pd.read_csv(name)
    #             ax.plot(data['epoch'], data["loss"], label='%s_train' % name.split("/")[-2])
    #             ax.plot(data['epoch'], data["val_loss"], label='%s_val' % name.split("/")[-2])
    #         plt.ylabel('Loss')
    #         plt.xlabel('Epoch')
    #         plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=fontsize)
    #         plt.tight_layout(rect=[0, 0, 0.75, 1])
    #         plt.savefig(os.path.join(dataname, "%s_losses.jpg" % net), dpi=300)
    #         plt.close()
