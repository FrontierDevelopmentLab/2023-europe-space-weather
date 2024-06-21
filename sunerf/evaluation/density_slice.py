import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader

parser = argparse.ArgumentParser(description='Evaluate density cube')
parser.add_argument('--ckpt_path', type=str, help='path to the SuNeRF checkpoint')
parser.add_argument('--result_path', type=str, help='path to the result directory')
args = parser.parse_args()

chk_path = args.ckpt_path
result_path = args.result_path
os.makedirs(result_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
n_points = 100
resolution = 512
max_radius = 100 / loader.Rs_per_ds
min_radius = (0.1 * u.AU).to_value(u.solRad) / loader.Rs_per_ds

densities = []
for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_points)), total=n_points):
    # DENSITY SLICE
    time = loader.normalize_datetime(timei)

    query_points_npy = np.stack(np.meshgrid(
        np.linspace(-max_radius, max_radius, resolution),
        np.linspace(-max_radius, max_radius, resolution),
        [0],
        [time]), -1).astype(np.float32)

    radius = np.sqrt(np.sum(query_points_npy[:, :, 0, 0, :2] ** 2, axis=-1))
    mask = (radius < min_radius) | \
           (radius > max_radius)

    model_out = loader.load_coords(query_points_npy[:, :, 0, 0, :])
    density, velocity = model_out['density'], model_out['velocity']
    # apply mask
    density[mask] = np.nan
    velocity[mask] = np.nan
    # print(density.max(), density.min())
    densities += [density]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f'{timei}')
    im = ax.imshow(density, norm='log',
                   extent=[-max_radius, max_radius, -max_radius, max_radius], cmap='inferno', origin='lower',
                   vmin=1e1, vmax=1e3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='N$_e$ / cm$^3$')

    # overlay velocity vectors
    quiver_pos = query_points_npy[::16, ::16]  # block_reduce(query_points_npy, (8, 8, 1, 1, 1), np.mean)
    quiver_vel = velocity[::16, ::16]  # block_reduce(velocity, (8, 8, 1), np.mean)
    ax.quiver(quiver_pos[:, :, 0, 0, 0], quiver_pos[:, :, 0, 0, 1],
              quiver_vel[:, :, 0], quiver_vel[:, :, 1],
              scale=30000, color='white')
    fig.savefig(os.path.join(result_path, f'dens_slice_{i:03d}.jpg'), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(np.linalg.norm(velocity, axis=-1),
                   extent=[-max_radius, max_radius, -max_radius, max_radius], cmap='viridis', origin='lower',
                   vmin=250, vmax=1500)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Velocity Magnitude')
    fig.savefig(os.path.join(result_path, f'velocity_{i:03d}.jpg'), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(continuity_loss,
                   extent=[-max_radius, max_radius, -max_radius, max_radius], cmap='viridis', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Continuity Loss')
    fig.savefig(os.path.join(result_path, f'continuity_loss_{i:03d}.jpg'), dpi=100)
    plt.close(fig)

    velocity_norm = np.linalg.norm(velocity, axis=-1)
    print(
        f'{timei} --> Velocity: {np.nanmin(velocity_norm):.01f} - {np.nanmax(velocity_norm):.01f} km/s; {np.nanmean(velocity_norm):.01f} km/s')

    # total_mass = np.sum(density) * loader.R ** 2 * u.cm ** 3

# for i, density in tqdm(enumerate(np.gradient(densities, axis=0)), total=n_points):
#     fig = plt.figure()
#     v_min_max = np.max(np.abs(density))
#     plt.imshow(density, norm=SymLogNorm(1e11, vmin=-1e13, vmax=1e13), cmap='bwr')
#     plt.colorbar(label='$\Delta N_e$')
#     plt.axis('off')
#     fig.savefig(os.path.join(video_path_dens, f'dens_diff_{i:03d}.jpg'), dpi=300)
#     plt.close(fig)
