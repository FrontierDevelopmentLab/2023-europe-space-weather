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

base_path = '/mnt/ground-data/training/HAO_pinn_allviewpoint'
chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_density')

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
n_points = 40
os.makedirs(video_path_dens, exist_ok=True)

densities = []
for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_points)), total=n_points):
    # TEST FOR KNOWN LOCATION
    lati = 0
    loni = 272.686
    di = 214.61000061  # (* u.m).to(u.solRad).value

    # DENSITY SLICE
    time = normalize_datetime(timei)

    query_points_npy = np.stack(np.mgrid[-100:100, -100:100, 0:1, 1:2], -1).astype(np.float32)

    mask = np.sqrt(np.sum(query_points_npy[:, :, 0, 0, :2] ** 2, axis=-1)) < 21

    query_points = torch.from_numpy(query_points_npy)
    query_points[..., -1] = time

    # Prepare points --> encoding.
    enc_query_points = loader.encoding_fn(query_points.view(-1, 4))

    raw = loader.fine_model(enc_query_points)
    density = 10 ** (15 + raw[..., 0])
    velocity = torch.tanh(raw[..., 1:]) / 3 * 250 + 50

    density = density.view(query_points_npy.shape[:2]).cpu().detach().numpy()
    velocity = velocity.view(query_points_npy.shape[:2] + velocity.shape[-1:]).cpu().detach().numpy()
    velocity = velocity / 10 #* density[..., None] / 1e27 # scale to mass flux
    # apply mask
    density[mask] = np.nan
    velocity[mask] = np.nan
    # print(density.max(), density.min())
    densities += [density]

    fig = plt.figure()
    im = plt.imshow(density, norm='log', vmin=4e24, vmax=8e26, extent=[-100, 100, -100, 100], cmap='inferno')
    # overlay velocity vectors
    quiver_pos = block_reduce(query_points_npy, (4, 4, 1, 1, 1), np.mean)
    quiver_vel = block_reduce(velocity, (4, 4, 1), np.mean)
    plt.quiver(quiver_pos[:, :, 0, 0, 0], quiver_pos[:, :, 0, 0, 1],
               quiver_vel[:, :, 0], quiver_vel[:, :, 1],
               scale=5, color='white')
    plt.colorbar(im, label='$N_e$')
    plt.axis('off')
    fig.savefig(os.path.join(video_path_dens, f'dens_slice_{i:03d}.jpg'), dpi=100)
    plt.close(fig)

# for i, density in tqdm(enumerate(np.gradient(densities, axis=0)), total=n_points):
#     fig = plt.figure()
#     v_min_max = np.max(np.abs(density))
#     plt.imshow(density, norm=SymLogNorm(1e11, vmin=-1e13, vmax=1e13), cmap='bwr')
#     plt.colorbar(label='$\Delta N_e$')
#     plt.axis('off')
#     fig.savefig(os.path.join(video_path_dens, f'dens_diff_{i:03d}.jpg'), dpi=300)
#     plt.close(fig)
