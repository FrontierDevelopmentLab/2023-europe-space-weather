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


class VigilDataset(Dataset):
    def __init__(self, fts_item_list: list, image_transforms=None):
        """
        __init__

        Arguments:
            fts_item_list: list - List of tuples containing filenames to load at the time
        """
        self.fts_items = fts_item_list
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.fts_items)

    def __getitem__(self, idx):
        fts_file = self.fts_items[idx]

        img_data = fits.getdata(fts_file)
        # TODO: resize images to a uniform scale - 2048,2048
        # TODO: set this up for batches instead?

        arrays = [img_data, img_data.copy(), img_data.copy()]
        img_data_rgb = np.stack(arrays, axis=2).astype(
            np.float32
        )  # Can't use np.int16 - use float / 65535
        # print(img_data_rgb.shape)
        # print(fts_file.split("/")[-1])
        if self.image_transforms is not None:
            img_data_rgb = self.image_transforms(img_data_rgb)

        return img_data_rgb, fts_file.split("/")[-1]


class FitsDataModule(LightningDataModule):
    """
    FitsDataModule:
        Class incorporating the dataloaders for Fits data from Cor1, Cor2 of the SECCHI spacecraft combo.
        This class enables the use of a consistent selection of data splits, saving the precise data in its member variables.

        Recommended Methods:
            train_dataloader() -> Dataloader: Returns the dataloader for the selected training data in the module
            test_dataloader() -> Dataloader: Returns the dataloader for the selected testing data in the module
            val_dataloader() -> Dataloader: Returns the validation dataloader
    """

    def __init__(self, config: dict = None):
        """__init__

        Args:
            hparams (dict): Dictionary encompassing the parameters passed in
                Dictionary Keys:
                    "Batch Size" - Default 32: Batchsize to use in the dataloaders
                    "num workers" - Default: 4 - Number of workers for the dataset
                    "train percentage" - Default: 0.85 - Percentage of collected data from Cor1, Cor2 to be used for training.
                                                         Each File will have an associated integer on whether an event is present (1) or not (0)
                    "test percentage" - Default: 0.1 - Percentage of collected data from Cor1, Cor2 to be used for testing.
                                                      Each File will have an associated integer on whether an event is present (1) or not (0)
                    "val percentage" - Default: 0.05 - Percentage of collected data from Cor1, Cor2 to be used for validation.
                                                       Each File will have an associated integer on whether an event is present (1) or not (0)

        """
        super().__init__()
        logging.info("Load data")

        # TODO note previous data_loader also loaded cme label

        self.config = config

        self.train_pct = config["train_pct"]
        self.valid_pct = config["valid_pct"]
        self.test_pct = 1 - (self.train_pct + self.valid_pct)
        if (self.train_pct + self.valid_pct + self.test_pct) != 1:
            print("Percentages do not add to 1!")

        self.num_workers = config["num_workers"]

        # need to ensure all images have the same resolution
        self.required_shape = (config["img_size"], config["img_size"])

        # TODO instead of resize check/discard images with incorrect resolution
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.required_shape, antialias=True),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _get_filelist(self):
        files = pd.read_csv(self.config["input_fnames_list"])["0"].values
        return files

    def setup(self, stage: str):
        """setup
        Method setting up the internals of the system
        including loading the relevant data (filenames) - Files have to be loaded in at runtime of the dataloader
        Dataloaders are created at the end of the method

        Args:
            stage (str): Stage for the data, encapsulating what this configuration is for - Example: Training, Testing, Validation
                         required by pytorch lightning
        """

        fnames = self._get_filelist()
        print("LEN FILES", fnames.shape)

        fnames_train, fnames_val, fnames_test = random_split(
            fnames, [self.train_pct, self.valid_pct, self.test_pct]
        )
        self.train_data = VigilDataset(
            fnames_train, image_transforms=self.image_transforms
        )
        self.test_data = VigilDataset(
            fnames_test, image_transforms=self.image_transforms
        )
        self.val_data = VigilDataset(fnames_val, image_transforms=self.image_transforms)
        print("Datasets set up")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, num_workers=None) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.test_data,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )

    def test_dataloader(self, num_workers=None) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.val_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )
