import glob
import os

files = glob.glob("./data/orl_faces/**/*.pgm")
for file in files:
    os.rename(
        file,
        file.replace(".pgm", ".png"))
