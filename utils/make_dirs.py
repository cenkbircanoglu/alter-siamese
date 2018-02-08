import os

from config import get_config


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_dirs():
    for path in [get_config().result_dir]:
        create_dir(path)
