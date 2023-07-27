import os

import numpy as np
import pandas as pd
import torch
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime

base_path = '/mnt/training/HAO_pinn_2viewpoints_backgrounds'
# base_path = '/mnt/ground-data/training/HAO_pinn_allviewpoint/'
chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_density_cube')

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
n_points = 40
os.makedirs(video_path_dens, exist_ok=True)

densities = []
r = np.linspace(21, 200, 256)
ph = np.linspace(-np.pi-np.pi/128, np.pi+np.pi/128, 258)
rr, phph = np.meshgrid(r, ph, indexing = "ij")
theta = (0.32395396 + 2.8176386) / 2 
# observer_offset = np.deg2rad(134.5641147025687)
# observer_offset = np.deg2rad(253)
observer_offset = np.deg2rad(180)
# observer_offset = 0

for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_points)), total=n_points):

    # DENSITY CUBE SLICE
    time = normalize_datetime(timei)

    # Load model to generate points
    # torch.save({'fine_model': sunerf.fine_model, 'coarse_model': sunerf.coarse_model,
    #            'wavelength': data_module.wavelength,
    #            'scaling_kwargs': sunerf.scaling_kwargs,
    #            'sampling_kwargs': sunerf.sampling_kwargs, 'encoder_kwargs': sunerf.encoder_kwargs,
    #            'test_kwargs': data_module.test_kwargs, 'config': config_data,
    #            'start_time': unnormalize_datetime(min(data_module.times)),
    #            'end_time': unnormalize_datetime(max(data_module.times))},
    #             save_path))

    # (x, y, z, t)
    #query_points_npy = np.array([[rr * np.cos(phph) * np.sin(theta),
    #                              rr * np.sin(phph) * np.sin(theta),
    #                              rr * np.cos(theta), 
    #                              time] for rr, phph in zip(r,ph)]).astype(np.float32)
    x = rr * np.cos(phph) * np.sin(theta)
    y = rr * np.sin(phph) * np.sin(theta)
    z = rr * np.cos(theta)
    t = np.ones_like(rr) * time
    query_points_npy = np.stack([x, y, z, t], -1).astype(np.float32)
    # (256, 258, 4)

    query_points = torch.from_numpy(query_points_npy)

    # Prepare points --> encoding.
    enc_query_points = loader.encoding_fn(query_points.view(-1, 4))

    raw = loader.fine_model(enc_query_points)
    density = raw[..., 0]
    velocity = raw[..., 1:]

    density = density.view(query_points_npy.shape[:2]).cpu().detach().numpy()
    velocity = velocity.view(query_points_npy.shape[:2] + velocity.shape[-1:]).cpu().detach().numpy()
    velocity = velocity / 10 #* density[..., None] / 1e27 # scale to mass flux

    # print(density.max(), density.min())
    # densities += [density]

    # fig = plt.figure()
    # im = plt.imshow(density.T, norm='log', vmin=4e24, vmax=8e26, extent=[-100,100,-100,100], cmap='inferno', origin='lower')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    im = ax.pcolormesh(phph - observer_offset, rr, density, edgecolors='face', norm='log', cmap="inferno", vmin=4e24, vmax=8e26)
    # overlay velocity vectors
    # quiver_pos = block_reduce(query_points_npy, (4, 4, 1), np.mean)
    # quiver_vel = block_reduce(velocity, (4, 4, 1), np.mean)
    # plt.quiver(quiver_pos[:, :, 0], quiver_pos[:, :, 1],
    #           quiver_vel[:, :, 0], quiver_vel[:, :, 1],
    #           scale=5, color='white')
    plt.colorbar(im, label='$N_e$')
    plt.axis('on')
    fig.savefig(os.path.join(video_path_dens, f'dens_cube_slice_{i:03d}.jpg'), dpi=100)
    plt.close(fig)

# for i, density in tqdm(enumerate(np.gradient(densities, axis=0)), total=n_points):
#     fig = plt.figure()
#     v_min_max = np.max(np.abs(density))
#     plt.imshow(density, norm=SymLogNorm(1e11, vmin=-1e13, vmax=1e13), cmap='bwr')
#     plt.colorbar(label='$\Delta N_e$')
#     plt.axis('off')
#     fig.savefig(os.path.join(video_path_dens, f'dens_diff_{i:03d}.jpg'), dpi=300)
#     plt.close(fig)
