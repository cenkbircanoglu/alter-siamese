import glob
import os

import pandas as pd

if __name__ == '__main__':
    names = glob.glob("/media/cenk/2TB1/alter_siamese/results/**/results.txt")
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
        data["name"] = data["network"] + " " + data["name"]
        data.drop("network", axis=1, inplace=True)
        data.sort_values("name").T.to_csv(
            os.path.join(os.path.dirname(os.path.dirname(name)), "%s_latex.xls" % name.split("/")[-2]))
