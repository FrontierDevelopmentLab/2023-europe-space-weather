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

    # def prepare_data(self):
    #     pass

    def setup(self, stage: str):
        self.current_stage = stage
        cor1_data = glob.glob("{}/*.fts".format(self.cor1_data_dir))  # inner corona
        cor2_data = glob.glob("{}/*.fts".format(self.cor2_data_dir))  # outer corona

        total_files = len(cor1_data + cor2_data)

        self.fits_test = -1
        self.fits_predict = -1
        fits_full = -1
        self.fits_train, self.fits_val = random_split(fits_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.fits_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fits_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fits_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.fits_predict, batch_size=self.batch_size)

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
