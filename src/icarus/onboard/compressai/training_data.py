# here decide what goes in yaml data_loader: data_dir: /mnt/onboard_data/training_data/v1


import glob
import logging
import multiprocessing
import os
from datetime import datetime, timedelta
from typing import Tuple

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import torch
import yaml
from astropy import units as u
from astropy.io.fits import getdata, getheader
from lightning.pytorch import LightningDataModule
from sunpy.map import Map
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm


def _get_filelist():
    """ """

    # option 1: get list of all files from directories
    # cor1_data = glob.glob("{}/*.fts".format(self.cor1_data_dir))  # inner corona
    # cor2_data = glob.glob("{}/*.fts".format(self.cor2_data_dir))  # outer corona
    # total_files = cor2_data if self.chosen_dataset == "cor2" else cor1_data

    # option 2: get list of files based on filtering from meta files
    # TODO hardcoded filenames
    cor1_meta_file = "/mnt/onboard_data/data/lists/meta_cor1.csv"
    cor2_meta_file = "/mnt/onboard_data/data/lists/meta_cor2.csv"
    df1 = pd.read_csv(cor1_meta_file)
    df2 = pd.read_csv(cor2_meta_file)

    # filter by resolution, doorstat and normal type observations
    # filtering by resolution is performed by looking at the x-coordinate of the centre pixel (not exactly half the resolution of the image)
    # TODO move vars to config
    res = 2048
    cor1_data = df1[
        (df1["DOORSTAT"] == 2)
        & (df1["obs_type"] == "n")
        & (df1["CRPIX1"] > res / 2 - 10)
        & (df1["CRPIX1"] < res / 2 + 10)
    ]["file_name"].values
    cor2_data = df2[
        (df2["DOORSTAT"] == 2)
        & (df2["obs_type"] == "n")
        & (df2["CRPIX1"] > res / 2 - 10)
        & (df2["CRPIX1"] < res / 2 + 10)
    ]["file_name"].values
    total_files = np.concatenate([cor1_data, cor2_data])  # TODO want cor1 or cor2?

    return total_files


if __name__ == "__main__":
    # TODO: add POLAR to meta data so can filter by angle

    fnames = _get_filelist()
    print(fnames[:10])

    df = pd.DataFrame(fnames)
    df[:100].to_csv("/mnt/onboard_data/training_data/fnames_test100.csv", index=False)
