#!/usr/bin/env python3

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inPlaceDir', type=str,
                        help="Directory to prune in-place.")
    args = parser.parse_args()

    exts = ["jpg", "png"]
    total_num = 0
    for subdir, dirs, files in os.walk(args.inPlaceDir):
        if subdir == args.inPlaceDir:
            continue
        nImgs = 0
        for fName in files:
            (imageClass, imageName) = (os.path.basename(subdir), fName)
            if any(imageName.lower().endswith("." + ext) for ext in exts):
                nImgs += 1
        total_num += nImgs
    print(total_num)
