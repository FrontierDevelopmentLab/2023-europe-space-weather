import glob
import logging
import multiprocessing
import os
from datetime import datetime, timedelta
from itertools import repeat
from warnings import simplefilter

import numpy as np
import torch
from astropy import units as u
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset
from tqdm import tqdm

from sunerf.train.coordinate_transformation import pose_spherical
from sunerf.train.ray_sampling import get_rays


class NeRFDataModule(LightningDataModule):

    def __init__(self, data_path_pB, data_path_tB, working_dir, Rs_per_ds, seconds_per_dt,
                 ref_time=None, radius_range=[21.5, 100],
                 batch_size=int(2 ** 10), num_workers=None, debug=False):
        super().__init__()
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt

        radius_range = np.array(radius_range) / Rs_per_ds
        print('Radius range ', radius_range)
        self.radius_range = radius_range

        logging.info('Load data')
        images, poses, rays, times, focal_lengths, wavelength = get_data(data_path_pB, data_path_tB, Rs_per_ds, debug)

        # normalize data
        self.ref_time = ref_time if ref_time is not None else times[0]
        times = np.array([normalize_datetime(t, seconds_per_dt, self.ref_time) for t in times])

        # keep data for logging
        self.images = images
        self.poses = np.array(poses)
        self.times = np.array(times)
        self.wavelength = wavelength

        N_GPUS = torch.cuda.device_count()
        self.batch_size = batch_size * N_GPUS
        self.points_batch_size = batch_size * N_GPUS

        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2

        # use an image of the first instrument
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

        self.valid_rays, self.valid_times, self.valid_images = valid_rays, valid_times, valid_images
        self.test_kwargs = {'focal': focal_lengths[test_idx],
                            'resolution': images[test_idx].shape[0]}

        # load all training rays
        rays, times, images = self._flatten_data([v for i, v in enumerate(rays) if i != test_idx],
                                                 [v for i, v in enumerate(times) if i != test_idx],
                                                 [v for i, v in enumerate(images) if i != test_idx])
        print("AFTER FLATTEN TRAIN", rays.shape, images.shape, times.shape)

        # remove nans (masked values) from data
        not_nan_pixels = ~np.any(np.isnan(images), -1)
        rays, times, images = rays[not_nan_pixels], times[not_nan_pixels], images[not_nan_pixels]

        # shuffle
        r = np.random.permutation(rays.shape[0])
        rays, times, images = rays[r], times[r], images[r]
        # save npy files
        # create file names
        logging.info('Save batches to disk')
        batch_file_rays = os.path.join(working_dir, 'rays_batches.npy')
        batch_file_times = os.path.join(working_dir, 'times_batches.npy')
        batch_file_images = os.path.join(working_dir, 'images_batches.npy')
        # save to disk
        np.save(batch_file_rays, rays)
        np.save(batch_file_times, times)
        np.save(batch_file_images, images)
        # to dataset
        self.train_rays, self.train_times, self.train_images = batch_file_rays, batch_file_times, batch_file_images

        logging.info('Create data sets')
        self.valid_data = TensorDataset({'rays': torch.tensor(self.valid_rays, dtype=torch.float32),
                                         'times': torch.tensor(self.valid_times, dtype=torch.float32),
                                         'images': torch.tensor(self.valid_images, dtype=torch.float32)})
        self.train_data = BatchesDataset(
            {'rays': self.train_rays, 'times': self.train_times, 'images': self.train_images},
            batch_size=self.batch_size)
        self.points_data = RandomCoordinateDataset(radius_range, [np.min(times), np.max(times)], self.points_batch_size)

    def _flatten_data(self, rays, times, images):
        flat_rays = np.concatenate(rays)
        flat_images = np.concatenate([i.reshape((-1, images[0].shape[-1])) for i in images])
        flat_time = np.concatenate([np.ones(i.shape[:2], dtype=np.float32).reshape((-1, 1)) * t
                                    for t, i in zip(times, images)])
        return flat_rays, flat_time, flat_images

    def train_dataloader(self):
        # handle batching manually
        ray_data_loader = DataLoader(self.train_data, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=True)
        points_loader = DataLoader(self.points_data, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.train_data, replacement=True,
                                                         num_samples=len(self.train_data)))
        return {'tracing': ray_data_loader, 'random': points_loader}

    def val_dataloader(self):
        # handle batching manually
        return DataLoader(self.valid_data, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                          shuffle=False)


class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = int(batch_size)

    def __len__(self):
        ref_file = list(self.batches_file_paths.values())[0]
        n_batches = np.ceil(np.load(ref_file, mmap_mode='r').shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        # lazy load data
        data = {k: np.copy(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, bf in self.batches_file_paths.items()}
        return data

    def clear(self):
        [os.remove(f) for f in self.batches_file_paths.values()]


class TensorDataset(Dataset):

    def __init__(self, tensor_dict, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.tensor_dict = tensor_dict
        self.batch_size = int(batch_size)

    def __len__(self):
        ref_tensor = list(self.tensor_dict.values())[0]
        return np.ceil(ref_tensor.shape[0] / self.batch_size).astype(int)

    def __getitem__(self, idx):
        data = {k: v[idx * self.batch_size: (idx + 1) * self.batch_size]
                for k, v in self.tensor_dict.items()}
        return data


class NumpyFilesDataset(Dataset):

    def __init__(self, *paths):
        self.paths = paths

    def __len__(self):
        return len(np.load(self.paths[0], mmap_mode='r'))

    def __getitem__(self, idx):
        return [np.copy(np.load(batch_file, mmap_mode='r')[idx]) for batch_file in self.paths]


def get_data(data_path_pB, data_path_tB, Rs_per_ds, debug=False):
    # HEO 1276 tB files, 1351 pB files
    # can assume matching names have matching obs times and angles
    s_maps_pB = sorted(glob.glob(data_path_pB))
    s_maps_tB = sorted(glob.glob(data_path_tB))

    if debug:
        s_maps_pB = s_maps_pB[::10]
        s_maps_tB = s_maps_tB[::10]

    with multiprocessing.Pool(os.cpu_count()) as p:
        data_pB = [d for d in tqdm(p.imap(_load_map_data, zip(s_maps_pB, repeat(Rs_per_ds))), total=len(s_maps_pB))]
        data_tB = [d for d in tqdm(p.imap(_load_map_data, zip(s_maps_tB, repeat(Rs_per_ds))), total=len(s_maps_tB))]

    _, poses, rays, times, focal_lengths = list(map(list, zip(*data_tB)))

    images = np.stack([[d[0] for d in data_tB], [d[0] for d in data_pB]], -1)

    ref_wavelength = Map(s_maps_tB[0]).wavelength.value if Map(s_maps_tB[0]).wavelength is not None else 0
    return images, poses, rays, times, focal_lengths, ref_wavelength


def _load_map_data(data):
    map_path, Rs_per_ds = data
    simplefilter('ignore')
    s_map = Map(map_path)
    # s_map = s_map.rotate(recenter=True, order=3)
    # compute focal length
    scale = s_map.scale[0]  # scale of pixels [arcsec/pixel]
    W = s_map.data.shape[0]  # number of pixels
    focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to_value(u.rad))

    time = s_map.date.datetime

    pose = pose_spherical(-s_map.heliographic_longitude.to(u.rad).value,
                          s_map.heliographic_latitude.to(u.rad).value,
                          s_map.dsun.to_value(u.R_sun) / Rs_per_ds).float().numpy()

    image = s_map.data.astype(np.float32)
    img_coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    all_rays = np.stack(get_rays(img_coords, pose), -2)

    # rays_o = all_rays[:, :, 0, :]
    # rays_d = all_rays[:, :, 1, :]
    # distance = np.linalg.norm(rays_o, axis=-1)
    # print("DISTANCE", distance.min(), distance.max())
    # print('Expected Distance', s_map.dsun.to_value(u.solRad))
    # print('Direction norm', np.linalg.norm(rays_d, axis=-1).min(), np.linalg.norm(rays_d, axis=-1).max())

    # crop to square
    min_dim = min(image.shape[:2])
    image = image[:min_dim, :min_dim]
    all_rays = all_rays[:min_dim, :min_dim]

    all_rays = all_rays.reshape((-1, 2, 3))

    return image, pose, all_rays, time, focal


def normalize_datetime(date, seconds_per_dt, ref_time):
    """Normalizes datetime object for ML input.

    Time starts at 2010-01-01 with max time range == 2 pi
    Parameters
    ----------
    date: input date

    Returns
    -------
    normalized date
    """
    return (date - ref_time).total_seconds() / seconds_per_dt


def unnormalize_datetime(norm_date: float, seconds_per_dt, ref_time) -> datetime:
    """Computes the actual datetime from a normalized date.

    Parameters
    ----------
    norm_date: normalized date

    Returns
    -------
    real datetime
    """
    return ref_time + timedelta(seconds=norm_date * seconds_per_dt)


class RandomCoordinateDataset(Dataset):

    def __init__(self, radius_range, time_range, batch_size):
        super().__init__()
        self.radius_range = radius_range  # [[inner, outer]]
        self.time_range = time_range  # [[start, end]]
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return 1

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 4).uniform_()
        r = self.radius_range[0] + (self.radius_range[1] - self.radius_range[0]) * random_coords[:, 0]
        theta = torch.pi * random_coords[:, 1]
        phi = 2 * torch.pi * random_coords[:, 2]
        t = self.time_range[0] + (self.time_range[1] - self.time_range[0]) * random_coords[:, 3]

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        coords = torch.stack([x, y, z, t], -1)

        return coords
