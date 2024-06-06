from typing import Tuple

import numpy as np
import torch
from astropy import units as u
from datetime import datetime
from sunpy.map.mapbase import PixelPair
from torch import nn
from torch.nn import DataParallel

from sunerf.train.coordinate_transformation import pose_spherical
from sunerf.model.model import PositionalEncoder
from sunerf.train.ray_sampling import get_rays
from sunerf.train.volume_render import nerf_forward
from sunerf.utilities.data_loader import normalize_datetime, unnormalize_datetime


class SuNeRFLoader:

    def __init__(self, state_path, resolution=None, focal=None, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        state = torch.load(state_path)
        self.resolution = resolution if resolution is not None else state['test_kwargs']['resolution']
        self.focal = focal if focal is not None else state['test_kwargs']['focal']
        self.wavelength = state['wavelength']
        self.times = state['times']
        self.config = state['config']
        self.device = device
        self.rho_model = state['model']['rho']
        self.v_model = state['model']['v']

        self.rho_model = DataParallel(self.rho_model).to(device)
        self.v_model = DataParallel(self.v_model).to(device)
        self.sampler = state['sampling']['stratified']
        self.hierarchical_sampler = state['sampling']['hierarchical']
        self.rendering = state['rendering']
        self.seconds_per_dt = state['seconds_per_dt']
        self.Mm_per_ds = state['Mm_per_ds']
        self.ref_time = state['ref_time']

    @property
    def start_time(self):
        return self.unnormalize_datetime(np.min(self.times))

    @property
    def end_time(self):
        return self.unnormalize_datetime(np.max(self.times))

    def load_observer_image(self, lat: float, lon: float, time: datetime,
                            distance: float = (1 * u.AU).to(u.solRad).value,
                            center: Tuple[float, float, float] = None, ref_pixel: PixelPair = None,
                            strides: int = 1, batch_size: int = 4096):
        with torch.no_grad():
            # convert to pose
            target_pose = pose_spherical(-lon, lat, distance, center).numpy()
            # load rays
            ref_pixel = PixelPair(x=(self.resolution - 1) / 2 * u.pix,
                                  y=(self.resolution - 1) / 2 * u.pix) if ref_pixel is None else ref_pixel
            rays_o, rays_d = get_rays(self.resolution, self.resolution, ref_pixel, self.focal, target_pose)
            rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)
            img_shape = rays_o[::strides, ::strides].shape[:2]
            rays_o = rays_o[::strides, ::strides].reshape([-1, 3]).to(self.device)
            rays_d = rays_d[::strides, ::strides].reshape([-1, 3]).to(self.device)

            time = normalize_datetime(time, self.seconds_per_dt)
            flat_time = (torch.ones(img_shape) * time).view((-1, 1)).to(self.device)
            # make batches
            rays_o, rays_d, time = torch.split(rays_o, batch_size), \
                                   torch.split(rays_d, batch_size), \
                                   torch.split(flat_time, batch_size)

            outputs = {}
            for b_rays_o, b_rays_d, b_time in zip(rays_o, rays_d, time):
                b_outs = nerf_forward(b_rays_o, b_rays_d, b_time,
                                      model=self.rho_model, sampler=self.sampler,
                                      hierarchical_sampler=self.hierarchical_sampler,
                                      rendering=self.rendering)
                for k in b_outs.keys():
                    if k not in outputs:
                        outputs[k] = []
                    outputs[k] += [b_outs[k].cpu()]
            outputs = {k: torch.cat(v).view(*img_shape, -1).numpy() for k, v in
                       outputs.items()}
            return outputs

    def normalize_datetime(self, time):
        return normalize_datetime(time, self.seconds_per_dt, self.ref_time)

    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_time)
