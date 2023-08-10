import glob
import logging
import multiprocessing
import os
from datetime import datetime, timedelta
from tqdm import tqdm

import numpy as np
import torch
from astropy import units as u
from pytorch_lightning import LightningDataModule
from sunpy.map import Map
from torch.utils.data import Dataset, DataLoader

from sunerf.train.coordinate_transformation import pose_spherical
from sunerf.train.ray_sampling import get_rays

from torch.utils.data import DataLoader, RandomSampler

from warnings import simplefilter

class NeRFDataModule(LightningDataModule):

    def __init__(self, hparams):
        super().__init__()

        logging.info('Load data')
        images, poses, rays, times, focal_lengths, wavelength = get_data(hparams)

        # fix time offset for transfer learning (TODO replace)
        if 'time_offset' in hparams:
            times = np.array(times) + hparams['time_offset']

        # keep data for logging
        self.images = images
        self.poses = np.array(poses)
        self.times = np.array(times)
        self.wavelength = wavelength

        self.working_dir = hparams['Training']['working_directory']
        os.makedirs(self.working_dir, exist_ok=True)

        N_GPUS = torch.cuda.device_count()
        self.batch_size = int(hparams['Training']['batch_size']) * N_GPUS
        self.points_batch_size = int(hparams['Training']['points_batch_size']) * N_GPUS

        num_workers = hparams['Training']['num_workers']
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2

        # use an image of the first instrument (SDO)
        test_idx = len(images) // 6

        print("BEFORE FLATTEN", np.array(rays).shape, np.array(times).shape, np.array(images).shape)
        valid_rays, valid_times, valid_images = self._flatten_data([v for i, v in enumerate(rays) if i == test_idx],
                                                                   [v for i, v in enumerate(times) if i == test_idx],
                                                                   [v for i, v in enumerate(images) if i == test_idx])
        print("AFTER FLATTEN", valid_rays.shape, valid_images.shape, valid_times.shape)                                                           
                                                                   
        # remove nans (masked values) from data
        # not_nan_pixels = np.any(~np.isnan(valid_images), -1) # instead of valid_images valid_times
        # print("NAN PIXELLING", not_nan_pixels.shape, not_nan_pixels[:3])
        # valid_rays, valid_times, valid_images = valid_rays[not_nan_pixels], valid_times[not_nan_pixels], valid_images[not_nan_pixels]
                                                                   
        # batching
        logging.info('Convert data to batches')
        n_batches = int(np.ceil(valid_rays.shape[0] / self.batch_size))
        print(valid_rays.shape, n_batches)
        valid_rays, valid_times, valid_images = np.array_split(valid_rays, n_batches), \
                                                np.array_split(valid_times, n_batches), \
                                                np.array_split(valid_images, n_batches)
        self.valid_rays, self.valid_times, self.valid_images = valid_rays, valid_times, valid_images
        self.test_kwargs = {'focal': focal_lengths[test_idx],
                            'resolution': images[test_idx].shape[0]}

        # load all training rays
        rays, times, images = self._flatten_data([v for i, v in enumerate(rays) if i != test_idx],
                                                 [v for i, v in enumerate(times) if i != test_idx],
                                                 [v for i, v in enumerate(images) if i != test_idx])

        # remove nans (masked values) from data
        not_nan_pixels = ~np.any(np.isnan(images), -1)
        rays, times, images = rays[not_nan_pixels], times[not_nan_pixels], images[not_nan_pixels]

        # shuffle
        r = np.random.permutation(rays.shape[0])
        rays, times, images = rays[r], times[r], images[r]
        # account for unequal length of last batch by padding with data from beginning
        pad = self.batch_size - rays.shape[0] % self.batch_size
        rays = np.concatenate([rays, rays[:pad]])
        times = np.concatenate([times, times[:pad]])
        images = np.concatenate([images, images[:pad]])
        # batching
        n_batches = rays.shape[0] // self.batch_size
        rays, times, images = np.array(np.split(rays, n_batches), dtype=np.float32), \
                              np.array(np.split(times, n_batches), dtype=np.float32), \
                              np.array(np.split(images, n_batches), dtype=np.float32)
        # save npy files
        # create file names
        logging.info('Save batches to disk')
        batch_file_rays = os.path.join(self.working_dir, 'rays_batches.npy')
        batch_file_times = os.path.join(self.working_dir, 'times_batches.npy')
        batch_file_images = os.path.join(self.working_dir, 'images_batches.npy')
        # save to disk
        np.save(batch_file_rays, rays)
        np.save(batch_file_times, times)
        np.save(batch_file_images, images)
        # to dataset
        self.train_rays, self.train_times, self.train_images = batch_file_rays, batch_file_times, batch_file_images

        logging.info('Create data sets')
        self.valid_data = BatchesDataset(self.valid_rays, self.valid_times, self.valid_images)
        self.train_data = NumpyFilesDataset(self.train_rays, self.train_times, self.train_images)
        self.points_data = RandomCoordinateDataset([21, 200], [np.min(times), np.max(times)], self.points_batch_size)

    def _flatten_data(self, rays, times, images):
        flat_rays = np.concatenate(rays)
        flat_images = np.concatenate([i.reshape((-1, images[0].shape[-1])) for i in images])
        flat_time = np.concatenate([np.ones(i.shape[:2], dtype=np.float32).reshape((-1, 1)) * t
                                    for t, i in zip(times, images)])
        return flat_rays, flat_time, flat_images

    def train_dataloader(self):
        # handle batching manually
        ray_data_loader = DataLoader(self.train_data, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        points_loader = DataLoader(self.points_data, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.train_data, replacement=True, num_samples=len(self.train_data)))
        return {'rays': ray_data_loader, 'points': points_loader}

    def val_dataloader(self):
        # handle batching manually
        return DataLoader(self.valid_data, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                          shuffle=False)


class BatchesDataset(Dataset):

    def __init__(self, *batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches[0])

    def __getitem__(self, idx):
        return [b[idx] for b in self.batches]


class NumpyFilesDataset(Dataset):

    def __init__(self, *paths):
        self.paths = paths

    def __len__(self):
        return len(np.load(self.paths[0], mmap_mode='r'))

    def __getitem__(self, idx):
        return [np.copy(np.load(batch_file, mmap_mode='r')[idx]) for batch_file in self.paths]


def get_data(config_data):
    debug = config_data['Debug']

    data_path_pB = config_data['data_path_pB']
    data_path_tB = config_data['data_path_tB']

    # HEO 1276 tB files, 1351 pB files
    # can assume matching names have matching obs times and angles
    s_maps_pB = sorted(glob.glob(data_path_pB))
    s_maps_tB = sorted(glob.glob(data_path_tB))

    if debug:
        s_maps_pB = s_maps_pB[::10]
        s_maps_tB = s_maps_tB[::10]

    with multiprocessing.Pool(os.cpu_count()) as p:
        data_pB = [d for d in tqdm(p.imap(_load_map_data, s_maps_pB), total=len(s_maps_pB))]
        data_tB = [d for d in tqdm(p.imap(_load_map_data, s_maps_tB), total=len(s_maps_tB))]
    
    _, poses, rays, times, focal_lengths = list(map(list, zip(*data_tB)))

    images = np.stack([[d[0] for d in data_tB], [d[0] for d in data_pB]], -1)

    ref_wavelength = Map(s_maps_tB[0]).wavelength.value if Map(s_maps_tB[0]).wavelength is not None else 0
    return images, poses, rays, times, focal_lengths, ref_wavelength


def _load_map_data(map_path):
    simplefilter('ignore')
    s_map = Map(map_path)
    # s_map = s_map.rotate(recenter=True, order=3)
    # compute focal length
    scale = s_map.scale[0]  # scale of pixels [arcsec/pixel]
    W = s_map.data.shape[0]  # number of pixels
    focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi / 180)

    time = normalize_datetime(s_map.date.datetime)

    print('COORDS', s_map.heliographic_longitude, s_map.heliographic_latitude)
    pose = pose_spherical(-s_map.heliographic_longitude.to(u.deg).value,
                          s_map.heliographic_latitude.to(u.deg).value,
                          s_map.dsun.to(u.solRad).value).float().numpy()

    image = s_map.data.astype(np.float32)
    image = image.T
    all_rays = np.stack(get_rays(image.shape[0], image.shape[1], s_map.reference_pixel, focal, pose), -2)

    # crop to square
    min_dim = min(image.shape[:2])
    image = image[:min_dim, :min_dim]
    all_rays = all_rays[:min_dim, :min_dim]

    all_rays = all_rays.reshape((-1, 2, 3))

    return image, pose, all_rays, time, focal


def normalize_datetime(date, max_time_range=timedelta(days=2)):
    """Normalizes datetime object for ML input.

    Time starts at 2010-01-01 with max time range == 2 pi
    Parameters
    ----------
    date: input date

    Returns
    -------
    normalized date
    """
    # TODO check this 30 days is valid for us
    return (date - datetime(2010, 1, 1)) / max_time_range * (2 * np.pi)


def unnormalize_datetime(norm_date: float) -> datetime:
    """Computes the actual datetime from a normalized date.

    Parameters
    ----------
    norm_date: normalized date

    Returns
    -------
    real datetime
    """
    return norm_date * timedelta(days=2) / (2 * np.pi) + datetime(2010, 1, 1)


class RandomCoordinateDataset(Dataset):

    def __init__(self, radius, times, batch_size):
        super().__init__()
        self.radius = radius # [[inner, outer]]
        self.times = times # [[start, end]]
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return 1

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 4).uniform_()
        r = self.radius[0] + (self.radius[1] - self.radius[0]) * random_coords[:, 0] ** 2 # sample more points closer to inner boundary
        theta = torch.pi * random_coords[:, 1]
        phi = 2 * torch.pi * random_coords[:, 2]
        t = self.times[0] + (self.times[1] - self.times[0]) * random_coords[:, 3]

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        coords = torch.stack([x, y, z, t], -1)

        return coords