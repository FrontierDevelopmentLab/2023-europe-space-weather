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

    def __init__(self, hparams: dict):
        """__init__

        Args:
            hparams (dict): Dictionary encompassing the parameters passed in
                Dictionary Keys:
                    "Batch Size" - Default 32: Batchsize to use in the dataloaders
                    "Data Directory Cor1" - Default: /mnt/onboard_data/data/cor1: Directory in which Cor1 FITS files are stored
                    "Data Directory Cor2" - Default: /mnt/onboard_data/data/cor2: Directory in which Cor2 FITS files are stored
                    "Data Directory Event" - Default: /mnt/onboard_data/data/events: Directory in which event files in the style produced by the "download_satellite_data" script are found
                    "train percentage" - Default: 0.85 - Percentage of collected data from Cor1, Cor2 to be used for training.
                                                         Each File will have an associated integer on whether an event is present (1) or not (0)
                    "test percentage" - Default: 0.1 - Percentage of collected data from Cor1, Cor2 to be used for testing.
                                                      Each File will have an associated integer on whether an event is present (1) or not (0)
                    "val percentage" - Default: 0.05 - Percentage of collected data from Cor1, Cor2 to be used for validation.
                                                       Each File will have an associated integer on whether an event is present (1) or not (0)
        """
        super().__init__()
        logging.info("Load data")

        self.batch_size = hparams.get("Batch Size", 32)  # default to 32

        cwd = os.getcwd()
        config_path = os.path.join(cwd, "..", "..", "..", "config")
        data_path = os.getcwd()
        with open(os.path.join(config_path, "onboard.yaml"), "r") as f:
            data_path = yaml.load(f, Loader=yaml.Loader)["drive_locations"]["datapath"]
        self.cor1_data_dir = hparams.get(
            "Data Directory Cor1", os.path.join(data_path, "data", "cor1")
        )
        self.cor2_data_dir = hparams.get(
            "Data Directory Cor2", os.path.join(data_path, "data", "cor2")
        )
        self.event_data_dir = hparams.get(
            "Data Directory Event", os.path.join(data_path, "data", "events")
        )
        self.p_train = hparams.get("train percentage", 0.85)
        self.p_test = hparams.get("test percentage", 0.1)
        self.p_val = hparams.get("validation percentage", 0.05)

    def setup(self, stage: str):
        """setup
            Method setting up the internals of the system
            including loading the relevant data (filenames and event IDs) - Files have to be loaded in at runtime of the dataloader

            Dataloaders are created at the end of the method
        Args:
            stage (str): Stage for the data, encapsulating what this configuration is for - Example: Training, Testing, Validation
        """
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
        all_fits_data = [
            self._load_fits_fileinfo_and_events(fname) for fname in total_files
        ]  # [self._load_fits_data_and_event(fname) for fname in total_files]

        self.fits_train, self.fits_val, self.fits_test = random_split(
            all_fits_data, [self.n_train, self.n_val, self.n_test]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.fits_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.fits_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.fits_test, batch_size=self.batch_size)

    def _load_fits_data_and_event(fname: str) -> Tuple[np.Array, int]:
        """_load_fits_data_and_events(fname)

            Loads the image data from a fits file, creating a 3D representation (through copies of the image data) in int16
            Also loads the list of known CMEs from files, checking if a CME is present at the same time as the image was taken, flagging it as such


        Args:
            fname (str): Filename to load

        Returns:
            Tuple[np.Array, int]: tuple of 3D array (w,h,n_c = 3) of image data, integer defining if CME is present (0 or 1)
        """
        img_data = fits.getdata(fname)
        header = fits.getheader(fname)
        img_time = (
            header["DATE-OBS"]
            .split(".")[0]
            .replace("T", "_")
            .replace("-", "_")
            .replace(":", "_")
            .split("_")
        )
        obs_time = datetime.strptime("_".join(img_time), "%Y_%m_%d_%H_%M_%S")
        found = False
        for f in glob.glob(self.event_data_dir):
            times = f.split("/")[-1].split(".")[0].split("_")
            start_time = datetime.strptime("_".join(times[1:7]), "%Y_%m_%d_%H_%M_%S")
            end_time = datetime.strptime("_".join(times[7:]), "%Y_%m_%d_%H_%M_%S")
            obs_in_file = (
                (not found) and obs_time >= start_time and obs_time <= end_time
            )
            if obs_in_file:
                events = pd.read_csv(f, header=0)
                for l in events:
                    event_start, event_end = l
                    event_start = (
                        event_start.split(".")[0]
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace(":", "_")
                        .split("_")
                    )
                    event_start = datetime.strptime(
                        "_".join(event_start), "%Y_%m_%d_%H_%M_%S"
                    )

                    event_end = (
                        event_end.split(".")[0]
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace(":", "_")
                        .split("_")
                    )
                    event_end = datetime.strptime(
                        "_".join(event_end), "%Y_%m_%d_%H_%M_%S"
                    )

                    obs_in_event = (
                        (not found)
                        and obs_time >= event_start
                        and obs_time <= event_end
                    )
                    if obs_in_event:
                        found = True

        # our input is gray-scaled; stack the same input three times to fake a rgb image
        arrays = [img_data, img_data.copy(), img_data.copy()]
        img_data_rgb = np.stack(arrays, axis=2).astype(np.int16)

        # normalise to [0, 1]
        # img_data_normalised = (img_data_rgb - img_data_rgb.min()) / (
        #     img_data_rgb.max() - img_data_rgb.min()
        # )
        # img_data_normalised = img_data_normalised.astype(np.float32)
        is_cme = int(found)
        return img_data_rgb, is_cme

    def _load_fits_fileinfo_and_events(fname: str) -> Tuple[str, int]:
        """_summary_

            Loading all fits image information in one go will overload the system -
            load filenames and information on whether there is an event instead, load in situ?

        Args:
            fname (str): filename to load

        Returns:
            Tuple[str, int]: tuple defining the filename and the event existence (0 or 1)
        """
        header = fits.getheader(fname)
        img_time = (
            header["DATE-OBS"]
            .split(".")[0]
            .replace("T", "_")
            .replace("-", "_")
            .replace(":", "_")
            .split("_")
        )
        obs_time = datetime.strptime("_".join(img_time), "%Y_%m_%d_%H_%M_%S")
        found = False
        for f in glob.glob(self.event_data_dir):
            times = f.split("/")[-1].split(".")[0].split("_")
            start_time = datetime.strptime("_".join(times[1:7]), "%Y_%m_%d_%H_%M_%S")
            end_time = datetime.strptime("_".join(times[7:]), "%Y_%m_%d_%H_%M_%S")
            obs_in_file = (
                (not found) and obs_time >= start_time and obs_time <= end_time
            )
            if obs_in_file:
                events = pd.read_csv(f, header=0)
                for l in events:
                    event_start, event_end = l
                    event_start = (
                        event_start.split(".")[0]
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace(":", "_")
                        .split("_")
                    )
                    event_start = datetime.strptime(
                        "_".join(event_start), "%Y_%m_%d_%H_%M_%S"
                    )

                    event_end = (
                        event_end.split(".")[0]
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace(":", "_")
                        .split("_")
                    )
                    event_end = datetime.strptime(
                        "_".join(event_end), "%Y_%m_%d_%H_%M_%S"
                    )

                    obs_in_event = (
                        (not found)
                        and obs_time >= event_start
                        and obs_time <= event_end
                    )
                    if obs_in_event:
                        found = True

        is_cme = int(found)
        return fname, is_cme
