"""
        Pinnerf Error Evaluation
        Evaluates the standard deviation in velocity and density for a given target box, looking for similar runtimes on the same base drive the differences in evaluation of velocity and density.
        
"""

import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime
import cv2


import imageio #gifs
import logging
import sys
import glob
from scipy.spatial.transform import Rotation #Used to Recalculate L5


# Unlike the data cube model, this time we will want to use alpha based on errors - ie, we want the plots to highlight where we can see the data reliably. If this does not show the CME, we have an issue...
def showcase_datacube_with_errors(masked_cube:np.ndarray, errors_cube:np.ndarray, global_min:float, global_max:float, tag:str, idx:int, x_fil:np.ndarray, y_fil:np.ndarray,z_fil:np.ndarray, alpha_expon:float, norm:str = "linear",fname_subtag = None):
        """    
                Creates a plot of the datacube that is passed in (expecting either speed or density), applying alpha based on the error cube added in.
                This means that in the end, the pixels that do turn up will be those the system is "sure" about, depending on the parameters set.

        Args:
            masked_cube (np.ndarray): Either a speed or density data cube
            errors_cube (np.ndarray): The cube of standard deviations associated with the previous one
            global_min (float): Global minimum value of that type for the original generation (not from the error array)
            global_max (float): Global maximum value of that type for the original generation (not from the error array)
            tag (str): Tag used for filename generation
            idx (int): Timestep idx
            x_fil (np.ndarray): points filtered for error acceptance
            y_fil (np.ndarray): points filtered for error acceptance
            z_fil (np.ndarray): points filtered for error acceptance
            alpha_expon (float): exponent applied to the normalized cube, setting how points will show up in the plot
            norm (str, optional): Sets what scaling is applied for the colormap (linear or logarithmic). Defaults to "linear".
            fname_subtag (str, optional): Addition to the figurename. Defaults to None.
        """
        #Calculate Alpha
        normalized_error_cube = (errors_cube - errors_cube.min())/(errors_cube.max() - errors_cube.min())
        sig_arg = normalized_error_cube**alpha_expon #Taken out the distance requirement from the sun (no r^2 masking).
        alpha = np.tanh(sig_arg) #Sigmoid or Tanh, function that yields range 0-1.
        # Setup plot
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        if(len(x_fil) and len(y_fil) and len(z_fil)):
                ax.scatter(x_fil, y_fil, z_fil, c=masked_cube, marker='.',norm=norm , vmin=global_min, vmax=global_max, alpha = alpha) #
                if norm == "log":
                        ticks = np.linspace(np.log(global_min), np.log(global_max), 10, endpoint = True)
                else:
                        ticks= np.linspace(global_min,global_max,10, endpoint = True)
                cbar = plt.colorbar(ax.collections[0], ax=ax, ticks = ticks)
                cbar.set_label('{}'.format(tag))
        ax.set_xlim(-250,250)
        ax.set_ylim(-250,250)
        ax.set_zlim(-250,250)
        ax.set_xlabel('X[ Solar Radii ]')
        ax.set_ylabel('Y[ Solar Radii ]')
        ax.set_zlabel('Z[ Solar Radii ]')
        ax.set_title('3D Scatter Plot of points based on {} at timestep {}'.format(tag, idx))
        ax.plot_surface(x_sun, y_sun, z_sun, color='gold', alpha=0.5)
        # Plot the Earth
        ax.plot_surface(x_earth, y_earth, z_earth, color='cyan', alpha=1)
        # Plot Earth Orbit
        ax.scatter(x_orbit, y_orbit, z_orbit, c='gray', marker='.', alpha = 0.1)    
        # Plot L5
        ax.plot_surface(x_l5, y_l5, z_l5, color = "red", alpha = 1)
        filename = os.path.join(video_path, f'{tag}_cube_{idx:03d}.jpg')

        if fname_subtag is not None:
                filename = os.path.join(video_path, f'{tag}_{fname_subtag}_cube_{idx:03d}.jpg')
        fig.savefig(filename, dpi=100)
        return filename



# Set up the folders

base_path = '/mnt/training/' 
base_path_model = os.path.join(base_path, 'HAO_pinn_2view_continuity')
chk_path = os.path.join(base_path_model, 'save_state.snf')
video_path = os.path.join(base_path_model, 'error_cubes')
os.makedirs(video_path, exist_ok=True)
additional_model_paths = glob.glob(os.path.join(base_path, 'HAO_pinn_2view_continuity*')) #Base model path with _1,2,3,4 amended...
checkpoints = [os.path.join(p, "save_state.snf") for p in additional_model_paths]
checkpoints.sort()
# Load Sunnerf
#loader = SuNeRFLoader(chk_path, resolution=512)


# Points in R_solar
num_points = 64
epsilon = 1e-7
percentile = 95
distance_mask = 21
maximum_distance_mask = 216 # Solar Radii - 1AU = ~215 S_/odot, so this means we restrict to 1AU
maximum_distance = 250
n_cubes = 70 # number of timesteps

x = np.linspace(-maximum_distance,maximum_distance,num_points)
y = np.linspace(-maximum_distance,maximum_distance,num_points)
z = np.linspace(-maximum_distance,maximum_distance,num_points)
xx,yy,zz = np.meshgrid(x,y,z,indexing = "ij")
solar_center = np.array([0,0,0])
distance = np.sqrt((xx - solar_center[0])**2 + (yy - solar_center[1])**2 + (zz - solar_center[2])**2)
#Cut out inner solar radii as per rest of the program

print("Masking inner {} Solar Radii, as well as anything beyond {} Solar Radii.".format(distance_mask, maximum_distance))
outside_sun_mask = distance > distance_mask
inside_earth_orbit_mask = distance < maximum_distance
x_filtered = xx[outside_sun_mask & inside_earth_orbit_mask]
y_filtered = yy[outside_sun_mask & inside_earth_orbit_mask]
z_filtered = zz[outside_sun_mask & inside_earth_orbit_mask]




# Calculate Earth Position and markers for plotting
mean_expected_velocity = np.array([-1.2207752, -2.5797436,  0 ], dtype=np.float32) #technically in solar radii per 2 days - ignore the days - taken from the data_cube_model analysis
earth_direction = mean_expected_velocity/np.linalg.norm(mean_expected_velocity) #This should be the direction of the earth on the xy plane, as the CME had been selected to hit the earth
earth_orbit_radius = 215.032
earth_location = earth_direction*earth_orbit_radius
l5_earth_angle = -np.pi/3
l5_location = np.matmul(Rotation.from_rotvec(np.asarray([0,0,l5_earth_angle])).as_matrix(), earth_location)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

marker_radius = 7
x_marker = np.outer(np.cos(u), np.sin(v))*marker_radius
y_marker = np.outer(np.sin(u), np.sin(v))*marker_radius
z_marker = np.outer(np.ones(np.size(u)), np.cos(v))*marker_radius

x_sun = np.outer(np.cos(u), np.sin(v))*distance_mask
y_sun = np.outer(np.sin(u), np.sin(v))*distance_mask
z_sun = np.outer(np.ones(np.size(u)), np.cos(v))*distance_mask

x_l5 = x_marker + l5_location[0]
y_l5 = y_marker + l5_location[1]
z_l5 = z_marker + l5_location[2]

x_earth = x_marker + earth_location[0]
y_earth = y_marker + earth_location[1]
z_earth = z_marker + earth_location[2]

x_orbit = earth_orbit_radius * np.cos(u)
y_orbit = earth_orbit_radius * np.sin(u)
z_orbit = np.zeros_like(u)

#List of cubes - populate with all found model options on the coords
v_cube_list = [] 
d_cube_list = []
s_cube_list = []

density_percentiles = []
speed_percentiles = []
min_max_speeds = []
min_max_densities = []

for checkpoint in checkpoints:
        # Create lists for each loader
        loader = SuNeRFLoader(checkpoint, resolution=512)
        densities = []
        velocities = []
        speeds = []
        for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_cubes)), total=n_cubes):    
                time = normalize_datetime(timei)
                t = np.ones_like(x_filtered) * time
                query_points_npy = np.stack([x_filtered, y_filtered, z_filtered, t], -1).astype(np.float32)
                # (256, 258, 4)

                query_points = torch.from_numpy(query_points_npy)
                enc_query_points = loader.encoding_fn(query_points.view(-1, 4))
                raw = loader.fine_model(enc_query_points)
                #electron_density = 10 ** (15 + x[..., 0])
                #velocity = torch.tanh(x[..., 1:]) / 3 * 250 + 50
                density = raw[...,0] # Function has been moved into the model, either remove it from the model or not.
                velocity = raw[..., 1:]

                if torch.isnan(density).any() or torch.isnan(velocity).any() or torch.isinf(density).any() or torch.isinf(velocity).any():
                        # remove nan values
                        density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
                        velocity = torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

                density = density.view(query_points_npy.shape[0]).cpu().detach().numpy()
                velocity = velocity.view(query_points_npy.shape[:1] + velocity.shape[-1:]).cpu().detach().numpy()
                speed = np.sqrt(velocity[...,0]**2+velocity[...,1]**2+velocity[...,2]**2)
                # velocity = velocity / 10
                densities.append(density)
                velocities.append(velocity)
                speeds.append(speed)
        densities = np.asarray(densities)
        velocities = np.asarray(velocities)
        speeds = np.asarray(speeds)
        
        perc_dens = np.percentile(densities,percentile)
        perc_speed = np.percentile(speeds,percentile)
        density_percentiles.append(perc_dens)
        speed_percentiles.append(perc_speed)
        
        min_max_densities.append([np.min(densities), np.max(densities)])
        min_max_speeds.append([np.min(speeds), np.max(speeds)])
        
        d_cube_list.append(densities)
        v_cube_list.append(velocities)
        s_cube_list.append(speeds)
d_cube_list = np.asarray(d_cube_list)
v_cube_list = np.asarray(v_cube_list)
s_cube_list = np.asarray(s_cube_list)

last_mask = None
model_mean_velocities = [] # (num_model, num_cubes, 3)
model_mean_densities = [] # (num_model, num_cubes, 1)
density_percentiles = np.asarray(density_percentiles)
speed_percentiles = np.asarray(speed_percentiles)
print(density_percentiles.shape)
#For each model, for each timestep, 
for model_index, (density_listing, velocity_listing, speed_listing) in enumerate(zip(d_cube_list, v_cube_list, s_cube_list)):
        d_percentile = density_percentiles[model_index]
        s_percentile = speed_percentiles[model_index]
        mean_densities = []
        mean_velocities = []
        for cube_index, (density, velocity, speed) in enumerate(zip(density_listing, velocity_listing, speed_listing)):
                density_mask = density > d_percentile
                speed_mask = speed > s_percentile
                density_and_speed_mask = density_mask & speed_mask
                if last_mask is not None:
                        # Last mask needs to be blurred - we want to remove voxels around the currently active points, as well as the points themselves
                        # last_mask exists, therefore we take out the background
                        last_mask = ~last_mask 
                        #Every spot that has been accepted last time is now disabled
                        # every new spot is still possible - achieving recent background subtraction
                        density_and_speed_mask = density_and_speed_mask & last_mask
                
                x_filtered_again = x_filtered[density_and_speed_mask]
                y_filtered_again = y_filtered[density_and_speed_mask]
                z_filtered_again = z_filtered[density_and_speed_mask]
                vmean = np.asarray([0,0,0])
                rho_mean = 0
                if len(velocity[density_and_speed_mask]):
                        vmean = velocity[density_and_speed_mask].mean(axis = 0)
                if len(density[density_and_speed_mask]):
                        rho_mean = density[density_and_speed_mask].mean(axis = 0)
                mean_velocities.append(vmean)
                mean_densities.append(rho_mean)        
                if last_mask is None:
                        last_mask = density_and_speed_mask
        mean_velocities = np.asarray(mean_velocities)
        mean_densities = np.asarray(mean_densities)
        model_mean_velocities.append(mean_velocities)
        model_mean_densities.append(mean_densities)
model_mean_densities = np.asarray(model_mean_densities)
model_mean_velocities = np.asarray(model_mean_velocities)

std_mean_density = 1e26*(model_mean_densities/1e26).std(axis = 0) # (n_cubes, 1) - standard deviation from selected models
std_mean_velocity = model_mean_velocities.std(axis = 0) # (n_cubes, 3)
