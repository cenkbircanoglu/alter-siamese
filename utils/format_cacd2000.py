import os
import shutil
from glob import glob

from utils.make_dirs import create_dir

files = glob(
    "/Users/cenk.bircanoglu/personal/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/CACD2000_by_age/*.jpg")

for filename in files:
    try:
        print(filename)
        splitted = filename.split("/")
        folder = "/".join(splitted[:-1])
        fname = splitted[-1]
        age = fname.split("_")[0]
        name = "%s_%s" % (fname.split("_")[1], fname.split("_")[2])
        new_folder = os.path.join(folder, age)
        create_dir(new_folder)
        new_filename = os.path.join(new_folder, fname)
        print(new_filename)
        shutil.move(filename, new_filename)
    except Exception as e:
        print(e)

files = glob(
    "/Users/cenk.bircanoglu/personal/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/CACD2000_by_name/*.jpg")
for filename in files:
    try:
        print(filename)
        splitted = filename.split("/")
        folder = "/".join(splitted[:-1])
        fname = splitted[-1]
        age = fname.split("_")[0]
        name = "%s_%s" % (fname.split("_")[1], fname.split("_")[2])
        new_folder = os.path.join(folder, name)
        create_dir(new_folder)
        new_filename = os.path.join(new_folder, fname)
        print(new_filename)
        shutil.move(filename, new_filename)
    except Exception as e:
        print(e)
