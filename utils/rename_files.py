import glob
import os

files = glob.glob("./data/dogs_cats/**/*.jpg")
for file in files:
    os.rename(
        file,
        file.replace(".pgm", ".png"))
