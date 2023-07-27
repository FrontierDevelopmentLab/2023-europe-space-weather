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


def generate_image(m, ref_map, img_fname, vmin=0, vmax=20):
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

    # imshow is mirror to m.plot
    plt.imshow(data, origin="lower", cmap="stereocor2", vmin=vmin, vmax=vmax)

    plt.savefig(img_data, dpi=100)
    plt.close()


def generate_video(video_path, video_name, framerate=10):
    """
    given directory of images, generate video
    given frames per second

    warning: if no write permissions it still pretends to work
    """
    images = sorted(glob(os.path.join(video_path, "*.jpg")))
    print(images)
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, framerate, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


def generate_images(fnames, target_path, req_polars=[0, 120, 240], prefix="cor2sa"):
    """
    generate 3x subsets of images per sequence for different POLAR angles
    """
    # check target path to output images exists
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # iterate over polarisation angles
    for i in range(0, len(req_polars)):
        req_polar = req_polars[i]
        polar_path = target_path + str(req_polar)

        # check path to output images exists
        if not os.path.exists(polar_path):
            os.makedirs(polar_path)

        # generate the background map
        ref_map = generate_ref_data(fnames, req_polar)

        # iterate over fnames
        for f in tqdm(fnames):
            m = Map(f)
            angle = m.meta["POLAR"]
            if angle == req_polar:
                img_fname = os.path.join(
                    polar_path,
                    prefix + "_" + os.path.basename(f).replace(".fts", ".jpg"),
                )
                generate_image(m, ref_map, img_fname, vmin=0, vmax=20)


if __name__ == "__main__":
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
        glob("/mnt/onboard_data/data/{}/{}_*_n*{}.fts").format(
            instrument, eventdate, satellite
        )
    )

    fgenerate_images(
        fnames_a,
        base_path,
        req_polars=angles,
        prefix="{}s{}".format(instrument, satellite),
    )
    for angle in angles:
        video_path = base_path + str(angle)
        video_name = os.path.join(video_path, "video_" + str(angle) + ".mp4")
        fgenerate_video(video_path, video_name, framerate=5)
