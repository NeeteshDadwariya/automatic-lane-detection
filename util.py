import glob
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from math import sqrt, degrees, atan, atan2
from dotmap import DotMap


def read_one_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img, os.path.basename(filename))


def read_images(filename=None, folder=None):
    folder = folder if folder is not None else "./data/original samples"
    filename = "{}/{}".format(folder, filename) if filename is not None else "{}/*.*".format(folder)
    return [read_one_image(file) for file in glob.glob(filename)]


def create_hls_from_hsl(h, s, l):
    return np.uint8([h / 2, 255 * l / 100, 255 * s / 100])

folder = './data/original samples'
filename = None
folder = folder if folder is not None else "./data/"
filename = "{}/{}".format(folder, filename) if filename is not None else "{}/*.*".format(folder)

for file, filename in read_images(folder=folder):
    print(filename)
#%%
