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

import imageio #gifs

base_path = '/mnt/training/HAO_pinn_cr_2view_a26978f_heliographic_reformat'

chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_cube')

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
os.makedirs(video_path_dens, exist_ok=True)
n_cubes = 70 # How many cubes to generate

# Points in R_solar
x = np.linspace(-250,250,32)
y = np.linspace(-250,250,32)
z = np.linspace(-250,250,32)

xx,yy,zz = np.meshgrid(x,y,z,indexing = "ij")
solar_center = np.array([0,0,0])
distance = np.sqrt((xx - solar_center[0])**2 + (yy - solar_center[1])**2 + (zz - solar_center[2])**2)
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

density_filenames = []
velocity_filenames = []

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
    density_filenames.append(os.path.join(video_path_dens, f'density_cube_{i:03d}.jpg'))
    '''
    # Plot velocity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_filtered, y_filtered, z_filtered, c=norm_mag, cmap = "inferno", alpha = norm_mag**1.5, marker='.') # norm_mag exponentiated for alpha channel
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot of points based on speed')
    # Add Sun
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sun = distance_mask * np.outer(np.cos(u), np.sin(v))
    y_sun = distance_mask * np.outer(np.sin(u), np.sin(v))
    z_sun = distance_mask * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the Sun
    ax.plot_surface(x_sun, y_sun, z_sun, color='gold', alpha=0.5)

    # Add Earth
    radius_earth = 0.009157683 # Solar radii = 6371 km - its a dot.
    x_earth = np.outer(np.cos(u), np.sin(v))*3# * radius_earth
    y_earth = np.outer(np.sin(u), np.sin(v))*3# * radius_earth
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))*3# * radius_earth

    # Earth position defined on x axis, 1 AU = 215.032 Solar Radii
    r_earth = 215.032
    center_earth = (-r_earth, 0, 0)
    # If center_earth = (-r_earth, 0, 0) then on a circle, this is at pi radians. If L5 is trailing, then L5 is at (r_earth*cos(2/3*np.pi), r_earth*sin(2/3*np.pi), 0)
    center_l5 = (r_earth*np.cos(2/3*np.pi), r_earth*np.sin(2/3*np.pi), 0)
    x_earth += center_earth[0]
    y_earth += center_earth[1]
    z_earth += center_earth[2]

    # Plot the Earth
    ax.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=1)
    # Plot L5
    x_l5 = x_earth + center_l5[0]
    y_l5 = x_earth + center_l5[1]
    z_l5 = x_earth + center_l5[2]
    ax.plot_surface(x_l5, y_l5, z_l5, color = "lightslategray", alpha = 1)


    # Plot earth orbit
    x_orbit = r_earth * np.cos(u)
    y_orbit = r_earth * np.sin(u)
    z_orbit = np.zeros_like(u)

    ax.scatter(x_orbit, y_orbit, z_orbit, c='gray', marker='.', alpha = 0.1)


    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('|v|')


    #plt.show()
    fig.savefig(os.path.join(video_path_dens, f'velocity_cube_{i:03d}.jpg'), dpi=100)
    velocity_filenames.append(os.path.join(video_path_dens, f'velocity_cube_{i:03d}.jpg'))


frame_duration = 0.5 #2fps
if len(density_filenames):
    with imageio.get_writer(os.path.join(video_path_dens,'density.gif'), mode='I', duration=frame_duration) as writer:
        for filename in density_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
if len(velocity_filenames):
    with imageio.get_writer(os.path.join(video_path_dens,'velocity.gif'), mode='I', duration=frame_duration) as writer:
        for filename in velocity_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)