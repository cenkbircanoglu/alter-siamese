# [age]_[gender]_[race]_[date&time].jpg

import os
import shutil
from glob import glob

from utils.make_dirs import create_dir

files = glob(
    "/Users/cenk.bircanoglu/personal/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/UTKface_inthewild_by_age/**/*.jpg")

for filename in files:
    try:
        print(filename)
        splitted = filename.split("/")
        folder = "/".join(splitted[:-2])
        fname = splitted[-1]
        age = fname.split("_")[0]
        gender = fname.split("_")[1]
        race = fname.split("_")[2]
        date_time = fname.split("_")[3]
        new_folder = os.path.join(folder, age)
        create_dir(new_folder)
        new_filename = os.path.join(new_folder, fname)
        print(new_filename)
        shutil.move(filename, new_filename)
    except Exception as e:
        print(e)
files = glob(
    "/Users/cenk.bircanoglu/personal/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/UTKface_inthewild_by_age_gender_race/**/*.jpg")

for filename in files:
    try:
        splitted = filename.split("/")
        folder = "/".join(splitted[:-2])
        fname = splitted[-1]
        age = fname.split("_")[0]
        gender = fname.split("_")[1]
        race = fname.split("_")[2]
        date_time = fname.split("_")[3]
        new_folder = os.path.join(folder, "%s_%s_%s" % (age, gender, race))
        create_dir(new_folder)
        new_filename = os.path.join(new_folder, fname)
        print(new_filename)
        shutil.move(filename, new_filename)
    except Exception as e:
        print(e)