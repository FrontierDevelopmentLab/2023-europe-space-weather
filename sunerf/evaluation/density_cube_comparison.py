import argparse
import glob
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import scipy
from dateutil.parser import parse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader

parser = argparse.ArgumentParser(description='Evaluate density cube')
parser.add_argument('--frames', type=str, help='path to the source frames')
parser.add_argument('--ckpt_path', type=str, help='path to the SuNeRF checkpoint')
parser.add_argument('--result_path', type=str, help='path to the result directory')
args = parser.parse_args()

frames = args.frames
chk_path = args.ckpt_path
result_path = args.result_path
os.makedirs(result_path, exist_ok=True)

# base time
date0 = parse("2010-04-03T09:04:00.000")

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)

densities = []
sunerf_densities = []
density_diffs = []
times = []

for fname in tqdm(sorted(glob.glob(frames))[24:25]):
    f_id = os.path.basename(fname).split('.')[0]

    o = scipy.io.readsav(fname)
    # o0 = scipy.io.readsav('/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/dens_stepnum_000.sav')
    # o1 = scipy.io.readsav('/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/dens_stepnum_001.sav')
    # print(o1['this_time'] - o0['this_time'])

    dt = date0 + timedelta(hours=float(o['this_time']))
    time = loader.normalize_datetime(dt)

    dens = o['dens']
    ph = o['ph1d']
    r = o['r1d']
    th = o['th1d']

    # clip radius to 100 Rsun
    mask = r < 100
    r = r[mask]
    dens = dens[:, :, mask]

    phi, theta, radius, t = np.meshgrid(ph + np.pi / 2, th, r, np.array([time]), indexing="ij")

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    query_points = np.stack([x, y, z, t], axis=-1, dtype=np.float32)
    query_points = query_points[:, :, :, 0]
    model_out = loader.load_coords(query_points)
    sunerf_density = model_out['density']

    # calibration
    model = LinearRegression(fit_intercept=False)
    dens_flat = dens.flatten()
    sunerf_dens_flat = sunerf_density.flatten()
    model.fit(sunerf_dens_flat.reshape(-1, 1), dens_flat)
    print(f'GT: {dens_flat.mean():.2E} SuNeRF: {sunerf_dens_flat.mean():.2E}; Coeff: {model.coef_[0]:.2E}')
    # sunerf_density = model.predict(sunerf_density.reshape(-1, 1)).reshape(sunerf_density.shape)

    # poly_fit = np.polyfit(sunerf_dens_flat, dens_flat, 1)
    # sunerf_density = sunerf_density * poly_fit[0]

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 5))

    rr, phph = np.meshgrid(r, ph, indexing="ij")

    ax = axs[0]
    z = np.transpose(dens[:, 64, :])
    pc = ax.pcolormesh(phph, rr, z, edgecolors='face', norm='log', cmap='inferno', vmin=1e1, vmax=1e3)
    fig.colorbar(pc, ax=ax,  label='Density [N$_e$ cm$^{-3}$]')

    ax.set_title("Ground-truth", va='bottom')

    ax = axs[1]
    z = np.transpose(sunerf_density[:, 64, :])
    pc = ax.pcolormesh(phph, rr, z, edgecolors='face', norm='log', cmap='inferno', vmin=1e1, vmax=1e3)
    fig.colorbar(pc, ax=ax,  label='Density [N$_e$ cm$^{-3}$]')
    ax.set_title("SuNeRF", va='bottom')

    # add title
    fig.suptitle(f"Time: {dt.isoformat(' ')}")

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'equatorial_slice_{f_id}.jpg'), dpi=300)
    plt.close('all')

    # vertical slice
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 5))

    rr, thth = np.meshgrid(r, th, indexing="ij")

    ax = axs[0]
    z = np.transpose(dens[64, :, :])
    pc = ax.pcolormesh(thth, rr, z, edgecolors='face', norm='log', cmap='inferno', vmin=1e1, vmax=1e3)
    fig.colorbar(pc, ax=ax, label='Density [N$_e$ cm$^{-3}$]')

    ax.set_title("Ground-truth", va='bottom')

    ax = axs[1]
    z = np.transpose(sunerf_density[64, :, :])
    pc = ax.pcolormesh(thth, rr, z, edgecolors='face', norm='log', cmap='inferno', vmin=1e1, vmax=1e3)
    fig.colorbar(pc, ax=ax, label='Density [N$_e$ cm$^{-3}$]')
    ax.set_title("SuNeRF", va='bottom')

    # add title
    fig.suptitle(f"Time: {dt.isoformat(' ')}")

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'polar_slice_{f_id}.jpg'), dpi=300)
    plt.close('all')

    dens_flat = np.log10(dens.flatten())
    sunerf_dens_flat = np.log10(sunerf_density.flatten())
    vmin, vmax = 1, 3
    mask = (sunerf_dens_flat > vmin) & (dens_flat > vmin) & (sunerf_dens_flat < vmax) & (dens_flat < vmax)
    sunerf_dens_flat = sunerf_dens_flat[mask]
    dens_flat = dens_flat[mask]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    h = ax.hist2d(dens_flat, sunerf_dens_flat, bins=100,
              cmap='viridis', vmin=0, vmax=2500, )
    # ax.scatter(dens_flat, sunerf_dens_flat, s=1)
    ax.set_xlabel('GT Density [log10 N$_e$ cm$^{-3}$]')
    ax.set_ylabel('SuNeRF Density [log10 N$_e$ cm$^{-3}$]')
    ax.set_aspect('equal')
    # plot polyfit
    x = np.linspace(vmin, vmax, 100)
    ax.plot(x, x, 'r--')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h[3], cax=cax, label='Counts')

    fig.tight_layout()

    fig.savefig(os.path.join(result_path, f'scatter_{f_id}.png'), dpi=300, transparent=True)
    plt.close('all')

    densities.append(dens.mean())
    sunerf_densities.append(sunerf_density.mean())
    diff = np.abs(dens - sunerf_density).mean()
    density_diffs.append(diff)
    times.append(dt)
    print(f'DIFF: {diff:.2E}')

    fig, axs = plt.subplots(2, 1, figsize=(5, 5))
    axs[0].imshow(dens.sum(axis=0), cmap='inferno', norm='log')
    axs[0].set_title('GT')
    axs[1].imshow(sunerf_density.sum(axis=0), cmap='inferno', norm='log')

    fig.savefig(os.path.join(result_path, f'sum_{f_id}.jpg'), dpi=300)
    plt.close('all')

densities = np.array(densities)
sunerf_densities = np.array(sunerf_densities)
density_diffs = np.array(density_diffs)
times = np.array(times)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(times, density_diffs, 'o-')
ax.set_title('Density difference')
ax.set_xlabel('Time')
ax.set_ylabel('Density difference')
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(result_path, 'density_diff.jpg'), dpi=300)
plt.close('all')

np.save(os.path.join(result_path, 'density_diff.npy'), density_diffs)
np.save(os.path.join(result_path, 'times.npy'), times)
