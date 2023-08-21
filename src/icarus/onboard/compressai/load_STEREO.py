"""
Authors: Emma + Hala

Read in sequence from given date/event
Read as 3 subsequences (observations are taken in 3 POLAR angles for reconstruction of singlepolarised image in later processing)
Save images with background subtraction and clipping
(Generate videos)
"""

import logging
import os
import re
import warnings
from datetime import date, datetime, timedelta
from functools import reduce
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import astropy
import astropy.io.fits as fits
import astropy.units as u
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sscws
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from rich.progress import Progress
from sunpy.map import Map
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm import tqdm


def generate_ref_data(filenames, polar_angle):
    """
    generate background reference image to be subtracted
    """

    ref_data = []
    for i in filenames:  # range(len(filenames)):
        # angle = fits.getheader(filenames[i])['POLAR']
        m = Map(i)
        angle = m.meta["POLAR"]
        # print(angle)
        if angle == polar_angle:
            ref_data.append(m.data / m.exposure_time.value)

    ref_map = np.mean(ref_data, axis=0)

    return ref_map


def load_image(m, ref_map, vmin=0, vmax=20):
    """
    for single image
    """
    # m = Map(f)

    pixel_coords = all_coordinates_from_map(m)
    solar_center = SkyCoord(0 * u.deg, 0 * u.deg, frame=m.coordinate_frame)
    pixel_radii = np.sqrt(
        (pixel_coords.Tx - solar_center.Tx) ** 2
        + (pixel_coords.Ty - solar_center.Ty) ** 2
    )
    # r2 masking
    mask = 1 - ((pixel_radii / pixel_radii.max()) ** 2) * 0.5
    mask = mask.value
    #mask[pixel_radii.value >= 0.9 * pixel_coords.Tx.max().value] = np.nan

    data = ((m.data / m.exposure_time.value) - ref_map) / mask

    return data #Normalize(vmin=vmin, vmax=vmax)(data)


class _Loader:
    def __init__(self, ref_map, angle):
        self.ref_map = ref_map
        self.angle = angle

    def _load(self, f):
        m = Map(f)
        map_angle = m.meta["POLAR"]
        if self.angle == map_angle:
            image = load_image(m, self.ref_map, vmin=0, vmax=20)
            time = m.date.datetime
            return image, time
        return None


def load_images(fnames, angle=0):
    """
    generate 3x subsets of images per sequence for different POLAR angles
    """
    # check target path to output images exists
    # generate the background map
    ref_map = generate_ref_data(fnames, angle)

    # iterate over fnames
    loader = _Loader(ref_map, angle)
    with Pool(os.cpu_count()) as p:
        res = [
            r
            for r in tqdm(p.map(loader._load, fnames), total=len(fnames))
            if r is not None
        ]
    images, times = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    return np.array(images), np.array(times)


def load_data(data_path, eventdate, angle=0):
    fnames = sorted(glob(data_path))
    images, times = load_images(fnames, angle)
    return images, times

def generate_images(fnames, base_path, req_polars):
    img_fn_list = []

    for angle in req_polars:
        angle_path = os.path.join(base_path, str(angle))
        os.makedirs(angle_path, exist_ok=True)
        images, times = load_images(fnames, angle)
        for image, time in tqdm(zip(images, times), total = len(images)):
            image_path = os.path.join(angle_path, str(time)+".npy")
            image_exp = np.expand_dims(image, 0)
            image_exp = np.tile(image_exp, (3,1,1))
            np.save(image_path, image_exp)
            img_fn_list.append(image_path)
    
    return img_fn_list


if __name__ == "__main__":
    # TODO add as argparse
    eventdate_list = np.load("./utils/eventdate_list.npy")
    #eventdates = ["20140222"]
    eventdates = eventdate_list#["20140222"]
    instruments = ["cor2"]  #'cor1'
    satellites = ["a"]  # "b"
    angles = [0, 120, 240] # polar angles

    complete_fnlist = []
    base_path = "/mnt/onboard_data/cai_prepped/"
    os.makedirs(base_path, exist_ok=True)
    for eventdate in eventdates:
        for instrument in instruments:
            for satellite in satellites:
                iter_path = os.path.join(base_path, "{}_{}_{}".format(
                eventdate, instrument, satellite))
                os.makedirs(iter_path, exist_ok=True)

                # set of files for given date, cor2, "normal", STEREO A (72 images)
                fnames_a = sorted(
                    glob(
                        "/mnt/onboard_data/data/{}/{}_*_n*{}.fts".format(
                            instrument, eventdate, satellite
                        )
                    )
                )
                if len(fnames_a) == 0:
                    continue
                rem_list = []
                for f in fnames_a:
                    image_data = fits.getdata(f, ext=0)
                    if image_data.shape != (2048,2048):
                        rem_list.append(f)
                for rem_item in rem_list:
                    fnames_a.remove(rem_item)
                iter_fnlist = generate_images(fnames_a, base_path, req_polars=angles)
                complete_fnlist.append(iter_fnlist)

    complete_fndf = pd.DataFrame(complete_fnlist)
    complete_fndf.T.to_csv(os.path.join(base_path, "fnlist.csv"), index=False)
