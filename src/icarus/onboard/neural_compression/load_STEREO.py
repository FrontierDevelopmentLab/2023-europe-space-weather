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
    for i in filenames[:15]:  # range(len(filenames)):
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
    mask[pixel_radii.value >= 0.9 * pixel_coords.Tx.max().value] = np.nan

    data = ((m.data / m.exposure_time.value) - ref_map) / mask

    return Normalize(vmin=vmin, vmax=vmax)(data)


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


if __name__ == "__main__":
    # TODO add as argparse
    eventdate = "20140222"
    instrument = "cor2"  #'cor1'
    satellite = "a"  # "b"

    base_path = "/mnt/onboard_data/visualization/cme_video_{}_{}/".format(
        instrument, satellite
    )
    angles = [0, 120, 240]  # polar angles
    os.makedirs(base_path, exist_ok=True)

    # set of files for given date, cor2, "normal", STEREO A (72 images)
    fnames_a = sorted(
        glob(
            "/mnt/onboard_data/data/{}/{}_*_n*{}.fts".format(
                instrument, eventdate, satellite
            )
        )
    )

    generate_images(
        fnames_a,
        base_path,
        req_polars=angles,
        prefix="{}s{}".format(instrument, satellite),
    )
    for angle in angles:
        video_path = base_path + str(angle)
        video_name = os.path.join(video_path, "video_" + str(angle) + ".mp4")
        generate_video(video_path, video_name, framerate=5)
