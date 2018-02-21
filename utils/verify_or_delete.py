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
import os

from PIL import Image


def verify_or_delete(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            fname = os.path.join(root, f)
            try:
                img = Image.open(fname)
                img.verify()
            except IOError as ie:
                print(fname)
                os.rmdir(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str,
                        help="Directory of images to partition in-place to 'train' and 'val' directories.")
    args = parser.parse_args()

    verify_or_delete(args.image_dir)
