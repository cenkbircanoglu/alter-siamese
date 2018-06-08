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
        print(name)
        try:
            data = pd.read_csv(name)
        except Exception as e:
            continue
        try:
            print(len(data))
            data.plot(x='epoch', y=["acc_metric", "val_acc_metric"], figsize=(9, 6), title=name.split("/")[-2])
            plt.legend(["train", "validation"], fontsize=fontsize)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.tight_layout(rect=[0, 0, 0.75, 1])
            plt.savefig(name.replace("logger.csv", "accuracy.jpg"), dpi=300)
            plt.close()
        except Exception as e:
            print(e)
        