import os

from config import Config


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_dirs():
    for path in [Config.result_dir]:
        create_dir(path)
