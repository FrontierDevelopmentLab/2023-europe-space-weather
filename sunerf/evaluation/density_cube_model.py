import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime

# base_path = '/mnt/training/HAO_pinn_cr_allview_a26978f_heliographic'
# observer_offset = np.deg2rad(90)

base_path = '/mnt/training/HAO_pinn_cr_2view_a26978f_heliographic_reformat'
observer_offset = np.deg2rad(90)

# base_path = '/mnt/training/HAO_pinn_2viewpoints_backgrounds'
# observer_offset = np.deg2rad(180)

chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_density_cube')

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
n_points = 70
os.makedirs(video_path_dens, exist_ok=True)

densities = []
r = np.linspace(21, 200, 256)
ph = np.linspace(-np.pi-np.pi/128, np.pi+np.pi/128, 258)
rr, phph = np.meshgrid(r, ph, indexing = "ij")
theta = (0.32395396 + 2.8176386) / 2 



for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_points)), total=n_points):

    # DENSITY CUBE SLICE
    time = normalize_datetime(timei)

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
    # density = raw[..., 0]
    density = 10 ** (15 + raw[..., 0])
    # velocity = raw[..., 1:]

    density = density.view(query_points_npy.shape[:2]).cpu().detach().numpy()
    # velocity = velocity.view(query_points_npy.shape[:2] + velocity.shape[-1:]).cpu().detach().numpy()
    # velocity = velocity / 10

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    im = ax.pcolormesh(phph - observer_offset, rr, density, edgecolors='face', cmap='viridis', norm='log', vmin=4e24, vmax=8e26)

    plt.colorbar(im, label='$N_e$')
    plt.axis('on')
    fig.savefig(os.path.join(video_path_dens, f'dens_cube_slice_{i:03d}.jpg'), dpi=100)
    plt.close(fig)
