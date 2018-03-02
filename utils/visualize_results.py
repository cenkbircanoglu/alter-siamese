import datetime
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


if __name__ == '__main__':
    names = glob.glob("../results/**/**/*.log")
    print(names)
    for name in names:
        print(name)
        data = pd.read_csv(name, delimiter="\t", names=["name", "train", "val", "test", "train_f", "val_f", "test_f"])
        data["network"] = data["name"].apply(lambda x: x.split(":")[-1].split("/")[-2])
        data["name"] = data["name"].apply(lambda x: x.split(":")[-1].split("/")[-1].replace("Loss", ""))
        data["train"] = data["train"].apply(lambda x: x.split(":")[-1]).apply(pd.to_numeric)
        data["val"] = data["val"].apply(lambda x: x.split(":")[-1]).apply(pd.to_numeric)
        data["test"] = data["test"].apply(lambda x: x.split(":")[-1]).apply(pd.to_numeric)
        data["train_f"] = data["train_f"].apply(lambda x: x.split(":")[-1]).apply(pd.to_numeric)
        data["val_f"] = data["val_f"].apply(lambda x: x.split(":")[-1]).apply(pd.to_numeric)
        data["test_f"] = data["test_f"].apply(lambda x: x.split(":")[-1]).apply(pd.to_numeric)
        data.to_csv(os.path.join(os.path.dirname(os.path.dirname(name)), "%s.xls" % name.split("/")[-2]))
        grouped = data.groupby('network')
        for network_name, group in grouped:
            fig = plt.figure()
            group[["name", "train", "val", "test"]].plot(kind="bar", x=["name"], rot=45)
            plt.legend(fontsize=8)
            plt.ylabel('Accuracy')
            plt.xlabel('Network-Loss')
            plt.tick_params(axis='both', which='minor', labelsize=8)
            plt.title(network_name.upper())
            plt.tight_layout(rect=[0, 0, 0.75, 1])
            plt.savefig(name.replace("results.txt", "%s_accuracy.jpg" % network_name), dpi=300)
            plt.close()
        grouped = data.groupby('network')
        for network_name, group in grouped:
            fig = plt.figure()
            group[["name", "train_f", "val_f", "test_f"]].plot(kind="bar", x=["name"], rot=45)
            plt.legend(fontsize=8)
            plt.ylabel('F1 Score')
            plt.xlabel('Network-Loss')
            plt.tick_params(axis='both', which='minor', labelsize=8)
            plt.title(network_name.upper())
            plt.tight_layout(rect=[0, 0, 0.75, 1])
            plt.savefig(name.replace("results.txt", "%s_f1.jpg" % network_name), dpi=300)
            plt.close()
