import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime
import cv2

def visualise_velocity(velocity, file_path):
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros(velocity.shape, dtype=np.uint8)
    hsv[..., 1] = 255
    mag = np.sqrt(velocity[...,0]**2+velocity[...,1]**2+velocity[...,2]**2)
    _, ang = cv2.cartToPolar(velocity[..., 0], velocity[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(file_path, bgr)
    return mag

# base_path = '/mnt/training/HAO_pinn_cr_allview_a26978f_heliographic'
# observer_offset = np.deg2rad(90)

base_path = '/mnt/training/HAO_pinn_cr_2view_a26978f_heliographic_reformat'
observer_offset = np.deg2rad(90)

chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_density_cube','video_cube')

# init loader
loader = SuNeRFLoader(chk_path, resolution=512, device= torch.device("cpu"))
n_points = 70
os.makedirs(video_path_dens, exist_ok=True)

densities = []
r = np.linspace(21, 200, 256)
ph = np.linspace(-np.pi-np.pi/128, np.pi+np.pi/128, 258)

theta = (0.32395396 + 2.8176386) / 2 

rr, phph = np.meshgrid(r, ph, indexing = "ij")


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

    raw = loader.fine_model.to("cpu")(enc_query_points.to("cpu")) # Force to CPU
    # raw = loader.fine_model(enc_query_points)
    # density = raw[..., 0]
    density = 10 ** (15 + raw[..., 0])
    velocity = raw[..., 1:]

    density = density.view(query_points_npy.shape[:2]).cpu().detach().numpy()
    velocity = velocity.view(query_points_npy.shape[:2] + velocity.shape[-1:]).cpu().detach().numpy()
    # velocity = velocity / 10

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    im = ax.pcolormesh(phph - observer_offset, rr, density, edgecolors='face', cmap='viridis', norm='log', vmin=2e24, vmax=8e28)
    plt.colorbar(im, label='$N_e$')
    plt.axis('on')
    fig.savefig(os.path.join(video_path_dens, f'dens_cube_slice_{i:03d}.jpg'), dpi=100)
    plt.close(fig)
    # Plot Magnitude of velocity
    mag = np.sqrt(velocity[...,0]**2+velocity[...,1]**2+velocity[...,2]**2)
    mag_n = (mag - mag.min())/(mag.max() - mag.min())
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    im = ax.pcolormesh(phph - observer_offset, rr, mag_n, edgecolors='face', cmap='viridis')
    plt.colorbar(im, label='$abs(V)$')
    plt.axis('on')
    fig.savefig(os.path.join(video_path_dens, f'velocity_cube_slice_{i:03d}.jpg'), dpi=100)
    plt.close(fig)
    