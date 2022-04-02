import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, degrees, atan, atan2
from dotmap import DotMap


def read_one_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img, filename.split("./data/")[1])


def read_images(filename=None):
    filename = "./data/{}".format(filename) if filename is not None else "./data/*.*"
    return [read_one_image(file) for file in glob.glob(filename)]


def create_hls_from_hsl(h, s, l):
    return np.uint8([h / 2, 255 * l / 100, 255 * s / 100])
