import os
import os.path as osp
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from os.path import basename, dirname, join
import time


def crop_z_axis(image, mask, expand):