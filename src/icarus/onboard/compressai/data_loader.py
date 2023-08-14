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
import seaborn as sns  
import matplotlib.pyplot as plt

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
from sunpy.map.maputils import all_coordinates_from_map
from astropy.coordinates import SkyCoord


class VigilDataset(Dataset):
    def __init__(self, fts_item_list: list, data_min, data_max, image_transforms=None):
        """
        __init__

        Arguments:
            fts_item_list: list - List of tuples containing filenames to load at the time
        """
        self.fts_items = fts_item_list
        self.data_min = data_min
        self.data_max = data_max
        self.image_transforms = image_transforms


    def __len__(self):
        return len(self.fts_items)
    
    def __getitem__(self, index):
        # read numpy file and return tuple (image, fname)
        fname = self.fts_items[index]
        img = np.load(fname)
        #print("overall data_min = ", self.data_min)
        #print("overall data_max = ", self.data_max)
        img_min = np.amin(img)
        img_max = np.amax(img)
        #img = (img-self.data_min)/(self.data_max-self.data_min)
        img = (img-img_min)/(img_max-img_min+1e-6)

        return torch.from_numpy(img).type(torch.FloatTensor), fname

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

        # Destination directory for preprocessed input images
        self.prep_path = config["prep_path"]

        # TODO instead of resize check/discard images with incorrect resolution
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.required_shape, antialias=True),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    
    def generate_ref_data(self, filenames, polar_angle):
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


    def generate_image(self, m, ref_map, img_fname, vmin=0, vmax=20):
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
        mask = 1 - ((pixel_radii / pixel_radii.max()) ** 2) * 0.5 # ranges 0.5-1
        mask = mask.value
        #print("mask before = ", mask)
        #print("are there NaNs = ", np.isnan(mask).any())
        #mask[pixel_radii.value >= 0.9 * pixel_coords.Tx.max().value] = np.nan
        #print("mask after = ", mask)
        data = ((m.data / m.exposure_time.value) - ref_map) / mask

        data_min = np.amin(data)
        data_max = np.amax(data)

        data = np.expand_dims(data, 0)
        # pretrained model has 3 channels (RGB)
        data = np.tile(data, (3,1,1))

        np.save(img_fname, data)
       
        # imshow is mirror to m.plot
        #plt.imshow(data, origin="lower", cmap="stereocor2", vmin=vmin, vmax=vmax)

        #plt.savefig(img_fname, dpi=100)
        #plt.close()
        return data_min, data_max


    def generate_images(self, fnames, target_path, req_polars=[0, 120, 240], prefix="cor2sa"):
        """
        generate 3x subsets of images per sequence for different POLAR angles
        """
        # check target path to output images exists
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        fname_list = []

        data_min, data_max = np.inf, -np.inf
        all_img_mins, all_img_maxs = [], []

        # iterate over polarisation angles
        for i in range(0, len(req_polars)):
            req_polar = req_polars[i]
            polar_path = target_path + str(req_polar)

            # check path to output images exists
            if not os.path.exists(polar_path):
                os.makedirs(polar_path)

            # generate the background map
            ref_map = self.generate_ref_data(fnames, req_polar)

            # iterate over fnames
            for f in tqdm(fnames):
                m = Map(f)
                angle = m.meta["POLAR"]
                if angle == req_polar:
                    img_fname = os.path.join(
                        polar_path,
                        prefix + "_" + "op_" + os.path.basename(f).replace(".fts", ".npy"),
                    )
                    img_min, img_max = self.generate_image(m, ref_map, img_fname, vmin=0, vmax=20)
                    fname_list.append(img_fname)

                    all_img_mins.append(img_min)
                    all_img_maxs.append(img_max)

                    #sns.boxplot(all_img_maxs)
                    #plt.savefig("seaborn_plot_max.png")
                    #sns.boxplot(all_img_mins)
                    #plt.savefig("seaborn_plot_min.png")

                    if img_min < data_min:
                        data_min = img_min
                    if img_max > data_max:
                        data_max = img_max
                    #data_min, data_max = -2000, 2000
        return fname_list, data_min, data_max#img_min, img_max#data_min, data_max

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
        prep_train_fnames, data_min, data_max = self.generate_images(fnames_train, self.prep_path)
        self.train_data = VigilDataset(
            prep_train_fnames, data_min, data_max, image_transforms=self.image_transforms
        )
        prep_test_fnames, _, _ = self.generate_images(fnames_test, self.prep_path)
        self.test_data = VigilDataset(
            prep_test_fnames, data_min, data_max, image_transforms=self.image_transforms
        )
        prep_val_fnames, _, _ = self.generate_images(fnames_val, self.prep_path)
        self.val_data = VigilDataset(prep_val_fnames, data_min, data_max, image_transforms=self.image_transforms)
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
            self.val_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )

    def test_dataloader(self, num_workers=None) -> DataLoader:
        num_workers = num_workers or self.num_workers
        return DataLoader(
            self.test_data,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )
