import os

import numpy as np
import pandas as pd
import torch
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.train.volume_render import jacobian
from sunerf.utilities.data_loader import normalize_datetime

base_path = '/glade/work/rjarolim/sunerf-cme/2view_v01'
chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_density')

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
n_points = 100
max_radius = 100
os.makedirs(video_path_dens, exist_ok=True)

densities = []
for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_points)), total=n_points):
    # DENSITY SLICE
    time = loader.normalize_datetime(timei)


    query_points_npy = np.stack(np.mgrid[-max_radius:max_radius, -max_radius:max_radius, 0:1, 1:2], -1).astype(np.float32)

    mask = (np.sqrt(np.sum(query_points_npy[:, :, 0, 0, :2] ** 2, axis=-1)) < 21) | \
           (np.sqrt(np.sum(query_points_npy[:, :, 0, 0, :2] ** 2, axis=-1)) > max_radius)

    query_points = torch.from_numpy(query_points_npy)
    query_points[..., -1] = time

    flat_query_points = query_points.reshape(-1, 4).to(loader.device)
    flat_query_points.requires_grad = True

    log_rho = loader.rho_model(flat_query_points)
    density = 10 ** log_rho
    velocity = loader.v_model(flat_query_points)

    jac_rho = jacobian(log_rho, flat_query_points)
    jac_v = jacobian(velocity, flat_query_points)
    jac_matrix = torch.cat([jac_rho, jac_v], dim=1)

    dlogRho_dx = jac_matrix[:, 0, 0]
    dVx_dx = jac_matrix[:, 1, 0]
    dVy_dx = jac_matrix[:, 2, 0]
    dVz_dx = jac_matrix[:, 3, 0]

    dlogRho_dy = jac_matrix[:, 0, 1]
    dVx_dy = jac_matrix[:, 1, 1]
    dVy_dy = jac_matrix[:, 2, 1]
    dVz_dy = jac_matrix[:, 3, 1]

    dlogRho_dz = jac_matrix[:, 0, 2]
    dVx_dz = jac_matrix[:, 1, 2]
    dVy_dz = jac_matrix[:, 2, 2]
    dVz_dz = jac_matrix[:, 3, 2]

    dlogRho_dt = jac_matrix[:, 0, 3]
    dVx_dt = jac_matrix[:, 1, 3]
    dVy_dt = jac_matrix[:, 2, 3]
    dVz_dt = jac_matrix[:, 3, 3]

    grad_logRho = torch.stack([dlogRho_dx, dlogRho_dy, dlogRho_dz], -1)
    div_V = (dVx_dx + dVy_dy + dVz_dz)
    v_dot_grad_Rho = (velocity * grad_logRho).sum(-1)
    continuity_eq = dlogRho_dt + div_V + v_dot_grad_Rho
    continuity_loss = continuity_eq.abs()

    density = density / (loader.Mm_per_ds * 1e8) ** 3 # Ne/ds^3 --> Ne/cm^3
    velocity = velocity * (loader.Mm_per_ds * 1e3) / loader.seconds_per_dt # km/s

    density = density.view(query_points_npy.shape[:2]).cpu().detach().numpy()
    velocity = velocity.view(query_points_npy.shape[:2] + velocity.shape[-1:]).cpu().detach().numpy()
    continuity_loss = continuity_loss.view(query_points_npy.shape[:2]).cpu().detach().numpy()
    # apply mask
    density[mask] = np.nan
    velocity[mask] = np.nan
    continuity_loss[mask] = np.nan
    # print(density.max(), density.min())
    densities += [density]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(density, norm='log', vmin=1e-10, vmax=1e-8,
                   extent=[-max_radius, max_radius, -max_radius, max_radius], cmap='inferno')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='N$_e$ / cm$^3$')

    # overlay velocity vectors
    quiver_pos = query_points_npy[::8, ::8]#block_reduce(query_points_npy, (8, 8, 1, 1, 1), np.mean)
    quiver_vel = velocity[::8, ::8]#block_reduce(velocity, (8, 8, 1), np.mean)
    ax.quiver(quiver_pos[:, :, 0, 0, 0], quiver_pos[:, :, 0, 0, 1],
               quiver_vel[:, :, 0], quiver_vel[:, :, 1],
               scale=10000, color='white')
    fig.savefig(os.path.join(video_path_dens, f'dens_slice_{i:03d}.jpg'), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(continuity_loss,
                   extent=[-max_radius, max_radius, -max_radius, max_radius], cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Continuity Loss')
    fig.savefig(os.path.join(video_path_dens, f'continuity_loss_{i:03d}.jpg'), dpi=100)
    plt.close(fig)

    print(f'{timei} --> Velocity: {np.nanmean(np.linalg.norm(velocity, axis=-1))}')

# for i, density in tqdm(enumerate(np.gradient(densities, axis=0)), total=n_points):
#     fig = plt.figure()
#     v_min_max = np.max(np.abs(density))
#     plt.imshow(density, norm=SymLogNorm(1e11, vmin=-1e13, vmax=1e13), cmap='bwr')
#     plt.colorbar(label='$\Delta N_e$')
#     plt.axis('off')
#     fig.savefig(os.path.join(video_path_dens, f'dens_diff_{i:03d}.jpg'), dpi=300)
#     plt.close(fig)
