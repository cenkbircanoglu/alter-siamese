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
            mean = data['timestamp'].values[0]#.pct_change().min()
            with open("../times_0.log", mode="a") as f:
                f.write("%s %s\n" % (name, mean))
        except Exception as e:
            print(name)
            os.remove(name)
            continue

