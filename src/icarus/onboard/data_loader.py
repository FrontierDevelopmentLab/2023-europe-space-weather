"""
    Dataloader.py

    Implement Pytorch dataloader for files found in the "Data" Subdirectory.
    Following https://lightning.ai/docs/pytorch/stable/data/datamodule.html

"""

import glob
import logging
import multiprocessing
import os
from datetime import datetime, timedelta

import astropy.io.fits as fits
import numpy as np
import torch
from astropy import units as u
from astropy.io.fits import getdata, getheader
from pytorch_lightning import LightningDataModule
from sunpy.map import Map
from torch.utils.data import DataLoader, Dataset, random_split


class FitsDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        logging.info("Load data")

        self.batch_size = hparams.get("Batch Size", 32)  # default to 32
        self.cor1_data_dir = hparams.get(
            "Data Directory Cor1", os.path.join(os.getcwd(), "Data", "Cor1")
        )
        self.cor2_data_dir = hparams.get(
            "Data Directory Cor2", os.path.join(os.getcwd(), "Data", "Cor2")
        )

        self.p_train = hparams.get("train percentage", 0.85)
        self.p_test = hparams.get("test percentage", 0.1)
        self.p_val = hparams.get("validation percentage", 0.05)

    def setup(self, stage: str):
        self.current_stage = stage
        cor1_data = glob.glob("{}/*.fts".format(self.cor1_data_dir))  # inner corona
        cor2_data = glob.glob("{}/*.fts".format(self.cor2_data_dir))  # outer corona

        n_total_files = len(cor1_data + cor2_data)
        # Should we set proportions?
        p_train = self.p_train
        p_test = self.p_test
        p_val = self.p_val

        self.n_train = int(n_total_files * p_train)
        self.n_test = int(n_total_files * p_test)
        self.n_val = int(n_total_files * p_val)

        n_train_val = self.n_train + self.n_val

        # There is a bit of time correlation here
        # - frankly, not important as long as timestamps are used to handle files, ie
        # can correlate timesteps to files, allowing us to reconstruct the correct physics
        total_files = cor1_data + cor2_data
        all_fits_data = [self._load_fits(fname) for fname in total_files]
        self.fits_train, self.fits_val, self.fits_test = random_split(
            all_fits_data, [self.n_train, self.n_val, self.n_test])

    def train_dataloader(self):
        return DataLoader(self.fits_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fits_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fits_test, batch_size=self.batch_size)

    def _load_fits(fname):
        img_data = fits.getdata(fname)

        # our input is gray-scaled; stack the same input three times to fake a rgb image
        arrays = [img_data, img_data.copy(), img_data.copy()]
        img_data_rgb = np.stack(arrays, axis=2).astype(np.int16)

        # normalise to [0, 1]
        # img_data_normalised = (img_data_rgb - img_data_rgb.min()) / (
        #     img_data_rgb.max() - img_data_rgb.min()
        # )
        # img_data_normalised = img_data_normalised.astype(np.float32)

        return img_data_rgb
