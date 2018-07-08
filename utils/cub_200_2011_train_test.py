import argparse
import errno
import os
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


def create_train_val_split(image_dir):
    images = {}
    with open('data/cub_200_2011/CUB_200_2011/images.txt', mode='r') as f:
        for i in f.readlines():
            values = i.replace('\n', '').split(' ')
            images[values[0]] = values[1]

    train_paths, test_paths = [], []
    with open('data/cub_200_2011/CUB_200_2011/train_test_split.txt', mode='r') as f:
        for i in f.readlines():
            values = i.replace('\n', '').split(' ')
            if eval(values[1]):
                train_paths.append(images.get(values[0]))
            else:
                test_paths.append(images.get(values[0]))

    for img in train_paths:
        orig_path = os.path.join(image_dir, 'images', img)
        new_dir = os.path.join(image_dir, 'train')
        new_path = os.path.join(image_dir, 'train', img)
        mkdir_p(new_dir)
        mkdir_p('/'.join(new_path.split('/')[:-1]))
        shutil.move(orig_path, new_path)

    for img in test_paths:
        orig_path = os.path.join(image_dir, 'images', img)
        new_dir = os.path.join(image_dir, 'test')
        new_path = os.path.join(image_dir, 'test', img)
        mkdir_p(new_dir)
        mkdir_p('/'.join(new_path.split('/')[:-1]))
        shutil.move(orig_path, new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/media/cenk/2TB1/alter_siamese/data/cub_200_2011/')
    args = parser.parse_args()

    create_train_val_split(args.image_dir)
