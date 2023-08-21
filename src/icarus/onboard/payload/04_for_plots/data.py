from astropy.io import fits
import numpy as np
from torchvision import transforms
import torch


OPTIONAL_LOG1 = False
OPTIONAL_LOG1 = True # ~ slightly better

def data_there_v1(x):
    # go from original data format to the desired range
    x = x.astype(np.float32)

    # (optionally)
    if OPTIONAL_LOG1:
        x = np.log1p(x).astype(np.float32) # inverse: back = np.expm1(Y)

    x_min = x.min()
    x_max = x.max()
    there = (x - x_min) / (x_max - x_min)  # 0 - 1
    return there, x_min, x_max

# def reconstr_orignal_range(rec, x_min, x_max):
#     b = (rec * (x_max - x_min)) + x_min
#
#     # (optionally)
#     if OPTIONAL_LOG1:
#         b = np.expm1(b)
#
#     return b

