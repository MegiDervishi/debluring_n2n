
import glob
import os

import PIL.Image
import numpy as np
from PIL import ImageOps

import ipdb
from collections import defaultdict

def load_image(fname):
    global format_stats, size_stats
    im = PIL.Image.open(fname)
    im = ImageOps.grayscale(im)
    return im

#parent_dir = "C:\\Users\\Marco\\Desktop\\n2n_deblur\\noise2noise\\datasets\\BSDS300-images\\BSDS300\\images"
parent_dir = "C:\\Users\\Marco\\Desktop\\n2n_deblur\\noise2noise\\datasets\\kodak"
#strings = ["train", "test"]

#for tt in strings:
input_dir = parent_dir #+ "\\" + tt
images = sorted(glob.glob(os.path.join(input_dir + "_c", '*.JPEG')))
images += sorted(glob.glob(os.path.join(input_dir + "_c", '*.jpg')))
images += sorted(glob.glob(os.path.join(input_dir + "_c", '*.png')))

for (idx, imgname) in enumerate(images):
    imgname = imgname.split('\\')[-1]
    image = load_image(input_dir + "_c\\" + imgname)
    image.save(input_dir + "\\" + imgname)