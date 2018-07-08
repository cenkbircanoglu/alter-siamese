#!/usr/bin/env bash


python utils/resize_images.py  --inputDir data/aloi_red2_ill/train --imsize 64
python utils/resize_images.py  --inputDir data/aloi_red2_ill/val --imsize 64
python utils/resize_images.py  --inputDir data/aloi_red2_ill/test --imsize 64
python utils/resize_images.py  --inputDir data/cacd2000_age/train --imsize 64
python utils/resize_images.py  --inputDir data/cacd2000_age/val --imsize 64
python utils/resize_images.py  --inputDir data/cacd2000_age/test --imsize 64
python utils/resize_images.py  --inputDir data/cats_dogs/train --imsize 64
python utils/resize_images.py  --inputDir data/cats_dogs/val --imsize 64
python utils/resize_images.py  --inputDir data/cats_dogs/test --imsize 64
python utils/resize_images.py  --inputDir data/gamo/train --imsize 64
python utils/resize_images.py  --inputDir data/gamo/val --imsize 64
python utils/resize_images.py  --inputDir data/gamo/test --imsize 64
python utils/resize_images.py  --inputDir data/utkface_age/train --imsize 64
python utils/resize_images.py  --inputDir data/utkface_age/val --imsize 64
python utils/resize_images.py  --inputDir data/utkface_age/test --imsize 64

python utils/resize_images.py  --inputDir data/books/train --imsize 224
python utils/resize_images.py  --inputDir data/books/val --imsize 224
python utils/resize_images.py  --inputDir data/books/test --imsize 224
python utils/resize_images.py  --inputDir data/cars_196/train --imsize 224
python utils/resize_images.py  --inputDir data/cars_196/val --imsize 224
python utils/resize_images.py  --inputDir data/cars_196/test --imsize 224
python utils/resize_images.py  --inputDir data/cub_200_2011/train --imsize 224
python utils/resize_images.py  --inputDir data/cub_200_2011/val --imsize 224
python utils/resize_images.py  --inputDir data/cub_200_2011/test --imsize 224
python utils/resize_images.py  --inputDir data/fashion/train --imsize 224
python utils/resize_images.py  --inputDir data/fashion/val --imsize 224
python utils/resize_images.py  --inputDir data/fashion/test --imsize 224
python utils/resize_images.py  --inputDir data/marvel/train --imsize 224
python utils/resize_images.py  --inputDir data/marvel/val --imsize 224
python utils/resize_images.py  --inputDir data/marvel/test --imsize 224
python utils/resize_images.py  --inputDir data/products/train --imsize 224
python utils/resize_images.py  --inputDir data/products/val --imsize 224
python utils/resize_images.py  --inputDir data/products/test --imsize 224