# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 17/11/2016 """

import os

from PIL import Image

__author__ = 'cenk'


def change_file_size(input_dir, desired_size):
    pairs_same, pairs_else, names = [], [], []
    for _, dirs, _ in os.walk(input_dir):
        for dir in dirs:
            subdirs = os.path.join(input_dir, dir)
            for _, subdir, files in os.walk(subdirs):
                for file in files:
                    if '.DS_Store' not in file:
                        try:
                            file_name = os.path.join(subdirs, file)
			    print(file_name)
                            im = Image.open(file_name)
			    old_size = im.size  # old_size[0] is in (width, height) format
                            ratio = float(desired_size)/max(old_size)
                            new_size = tuple([int(x*ratio) for x in old_size])

                            im = im.resize(new_size, Image.ANTIALIAS)
                            # create a new image and paste the resized on it
                            new_im = Image.new("RGB", (desired_size, desired_size))
                            new_im.paste(im, ((desired_size-new_size[0])//2,
                                 (desired_size-new_size[1])//2))
                            new_im.save(file_name)
                        except Exception as e:
                            print(e.message)
                            print('ERROR')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inputDir', type=str, help='an integer for the accumulator')
    parser.add_argument('--imsize', type=int, help='an integer for the accumulator')

    args = parser.parse_args()
    change_file_size(args.inputDir, args.imsize)
