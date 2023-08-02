import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime
import cv2

from mpl_toolkits.mplot3d import Axes3D

base_path = '/mnt/training/HAO_pinn_cr_2view_a26978f_heliographic_reformat'
observer_offset = np.deg2rad(90)

chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_cube')

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
os.makedirs(video_path_dens, exist_ok=True)
n_cubes = 70 # How many cubes to generate

# Points in R_solar
x = np.linspace(-100,100,32)
y = np.linspace(-100,100,32)
z = np.linspace(-100,100,32)

xx,yy,zz = np.meshgrid(x,y,z,indexing = "ij")
sphere_center = np.array([0,0,0])
distance = np.sqrt((xx - sphere_center[0])**2 + (yy - sphere_center[1])**2 + (zz - sphere_center[2])**2)
distance_mask = 21 #Cut out 21 solar radii as per rest of the program
outside_sphere_mask = distance > distance_mask
x_filtered = xx[outside_sphere_mask]
y_filtered = yy[outside_sphere_mask]
z_filtered = zz[outside_sphere_mask]

densities = [] #1d - Generates Density at each point in each cube
velocities = [] #3d - 3 Velocity at each point in each cube
speeds = [] #1d - Speed at each point in each cube

for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_cubes)), total=n_cubes):
    
    time = normalize_datetime(timei)
    t = np.ones_like(x_filtered) * time
    query_points_npy = np.stack([x_filtered, y_filtered, z_filtered, t], -1).astype(np.float32)
    # (256, 258, 4)

    query_points = torch.from_numpy(query_points_npy)
    enc_query_points = loader.encoding_fn(query_points.view(-1, 4))
    raw = loader.fine_model(enc_query_points)
    
    density = 10 ** (15 + raw[..., 0])
    velocity = raw[..., 1:]

    if torch.isnan(density).any() or torch.isnan(velocity).any() or torch.isinf(density).any() or torch.isinf(velocity).any():
        # remove nan values
        density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    density = density.view(query_points_npy.shape[0]).cpu().detach().numpy()
    velocity = velocity.view(query_points_npy.shape[:1] + velocity.shape[-1:]).cpu().detach().numpy()
    # velocity = velocity / 10
    mag = np.sqrt(velocity[...,0]**2+velocity[...,1]**2+velocity[...,2]**2)



    densities.append(density)
    velocities.append(velocity)
    speeds.append(mag)


global_max_v = np.asarray(speeds).max()
global_min_v = np.asarray(speeds).min()
global_max_rho = np.asarray(densities).max()
global_min_rho = np.asarray(densities).min()
print(global_max_rho, global_min_rho)

for i, (rho, v, abs_v) in enumerate(zip(densities, velocities, speeds)):
    #rho_norm = (rho - global_min_rho)/(global_max_rho - global_min_rho + 1e-20)
    norm_mag = (abs_v-global_min_v)/(global_max_v-global_min_v)
    plt.close("all")
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_filtered, y_filtered, z_filtered, c=rho, marker='.',norm='log', alpha = rho_norm, vmin=2e24, vmax=8e28)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot of points based on density')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('n_e')
    #plt.show()
    fig.savefig(os.path.join(video_path_dens, f'density_cube_{i:03d}.jpg'), dpi=100)
    '''
    # Plot velocity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_filtered, y_filtered, z_filtered, c=norm_mag, alpha = norm_mag, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot of points based on speed')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('|v|')
    #plt.show()
    fig.savefig(os.path.join(video_path_dens, f'velocity_cube_{i:03d}.jpg'), dpi=100)
    
