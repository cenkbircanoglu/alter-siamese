#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import errno
import os
import random
import shutil


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_imgs(image_dir):
    exts = ["jpg", "png"]

    # All images with one image from each class put into the validation set.
    all_imgs_m = []
    classes = set()
    val_imgs = []
    for subdir, dirs, files in os.walk(image_dir):
        for fName in files:
            (image_class, image_name) = (os.path.basename(subdir), fName)
            if any(image_name.lower().endswith("." + ext) for ext in exts):
                if image_class not in classes:
                    classes.add(image_class)
                    val_imgs.append((image_class, image_name))
                else:
                    all_imgs_m.append((image_class, image_name))
    print("+ Number of Classes: '{}'.".format(len(classes)))
    return (all_imgs_m, val_imgs)


def create_train_val_split(image_dir, val_ratio):
    print("+ Val ratio: '{}'.".format(val_ratio))

    (all_imgs_m, val_imgs) = get_imgs(image_dir)

    train_val_idx = int((len(all_imgs_m) + len(val_imgs)) * val_ratio) - len(val_imgs)
    assert (train_val_idx > 0)  # Otherwise, val_ratio is too small.

    random.shuffle(all_imgs_m)
    val_imgs += all_imgs_m[0:train_val_idx]
    train_imgs = all_imgs_m[train_val_idx:]

    print("+ Training set size: '{}'.".format(len(train_imgs)))
    print("+ Validation set size: '{}'.".format(len(val_imgs)))

    for person, img in train_imgs:
        orig_path = os.path.join(image_dir, person, img)
        new_dir = os.path.join(image_dir, 'train', person)
        new_path = os.path.join(image_dir, 'train', person, img)
        mkdir_p(new_dir)
        shutil.move(orig_path, new_path)

    for person, img in val_imgs:
        orig_path = os.path.join(image_dir, person, img)
        new_dir = os.path.join(image_dir, 'val', person)
        new_path = os.path.join(image_dir, 'val', person, img)
        mkdir_p(new_dir)
        shutil.move(orig_path, new_path)

    for person, img in val_imgs:
        d = os.path.join(image_dir, person)
        if os.path.isdir(d):
            os.rmdir(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str,
                        help="Directory of images to partition in-place to 'train' and 'val' directories.")
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help="Validation to training ratio.")
    args = parser.parse_args()

    create_train_val_split(args.image_dir, args.val_ratio)
