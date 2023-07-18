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
import yaml
from astropy import units as u
from astropy.io.fits import getdata, getheader
from lightning.pytorch import LightningDataModule
from sunpy.map import Map
from torch.utils.data import DataLoader, Dataset, random_split


class FitsDataModule(LightningDataModule):
    def __init__(self, hparams):
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
            "Data Directory Event",os.path.join(data_path, "data", "events")
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
            all_fits_data, [self.n_train, self.n_val, self.n_test]
        )

    def train_dataloader(self):
        return DataLoader(self.fits_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fits_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fits_test, batch_size=self.batch_size)

    def _load_fits(fname):
        img_data = fits.getdata(fname)
        header = fits.getheader(fname)
        img_time = header["DATE-OBS"].split(".")[0].replace("T","_").replace("-","_").replace(":","_").split("_")
        obs_time = datetime.strptime("_".join(img_time),"%Y_%m_%d_%H_%M_%S")
        found = 0
        for f in glob.glob(self.event_data_dir):
            times = f.split('/')[-1].split('.')[0].split('_')
            start_time = datetime.strptime("_".join(times[1:7]),"%Y_%m_%d_%H_%M_%S")
            end_time = datetime.strptime("_".join(times[7:]),"%Y_%m_%d_%H_%M_%S")
            obs_in_file = (!found and (obs_time >= start_time and obs_time <= end_time))
            if obs_in_file:
                events = pd.read_csv(f,header=0)
                for l in events:
                    event_start, event_end = l
                    event_start = event_start.split(".")[0].replace(" ","_").replace("-","_").replace(":","_").split("_")
                    event_start = datetime.strptime("_".join(event_start),"%Y_%m_%d_%H_%M_%S")

                    event_end = event_end.split(".")[0].replace(" ","_").replace("-","_").replace(":","_").split("_")
                    event_end = datetime.strptime("_".join(event_end),"%Y_%m_%d_%H_%M_%S")

                    obs_in_event = (!found and (obs_time>=event_start and obs_time<=event_end))
                    if obs_in_event:
                        found = 1

        # our input is gray-scaled; stack the same input three times to fake a rgb image
        arrays = [img_data, img_data.copy(), img_data.copy()]
        img_data_rgb = np.stack(arrays, axis=2).astype(np.int16)

        # normalise to [0, 1]
        # img_data_normalised = (img_data_rgb - img_data_rgb.min()) / (
        #     img_data_rgb.max() - img_data_rgb.min()
        # )
        # img_data_normalised = img_data_normalised.astype(np.float32)
        is_cme = found
        return img_data_rgb, is_cme
