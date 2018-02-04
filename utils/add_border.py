import glob

from PIL import Image

files = glob.glob("/Users/cenk.bircanoglu/Downloads/trashnet-data/dataset-copy/**/*.jpg")
for file in files:
    old_im = Image.open(file)
    old_size = old_im.size

    new_size = (256 * 4, 256 * 4)
    new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
    new_im.paste(old_im, ((new_size[0] - old_size[0]) / 2,
                          (new_size[1] - old_size[1]) / 2))

    new_im.show()
