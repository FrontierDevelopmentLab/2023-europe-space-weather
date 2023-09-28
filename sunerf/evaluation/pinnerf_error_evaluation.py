"""
        Pinnerf Error Evaluation
        Evaluates the standard deviation for a given target, looking for similar runtimes on the same base drive the differences in evaluation of velocity and density.
        This enables the script to create figures for the error density, that is, showcase the standard deviation around each point.
        
        Alongside this, a 3D plot is created that thresholds shown points by that error.
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


# Set up the folders

base_path = '/mnt/training/' 
base_path_model = os.path.join(base_path, 'HAO_pinn_2view_continuity')
chk_path = os.path.join(base_path_model, 'save_state.snf')
video_path = os.path.join(base_path_model, 'error_cubes')
os.makedirs(video_path, exist_ok=True)
additional_model_paths = glob.glob(os.path.join(base_path, 'HAO_pinn_2view_continuity_*')) #Base model path with _1,2,3,4 amended...
chk_additionals = [os.path.join(p, "save_state.snf") for p in additional_model_paths]
# Load Sunnerf
loader = SuNeRFLoader(chk_path, resolution=512)


# Points in R_solar
num_points = 64
epsilon = 1e-7
percentile = 95

x = np.linspace(-250,250,num_points)
y = np.linspace(-250,250,num_points)
z = np.linspace(-250,250,num_points)
xx,yy,zz = np.meshgrid(x,y,z,indexing = "ij")
solar_center = np.array([0,0,0])
distance = np.sqrt((xx - solar_center[0])**2 + (yy - solar_center[1])**2 + (zz - solar_center[2])**2)
#Cut out inner solar radii as per rest of the program
distance_mask = 21
maximum_distance = 216 # Solar Radii - 1AU = ~215 S_/odot, so this means we restrict to 1AU
print("Masking inner {} Solar Radii, as well as anything beyond {} Solar Radii.".format(distance_mask, maximum_distance))
outside_sun_mask = distance > distance_mask
inside_earth_orbit_mask = distance < maximum_distance
x_filtered = xx[outside_sun_mask & inside_earth_orbit_mask]
y_filtered = yy[outside_sun_mask & inside_earth_orbit_mask]
z_filtered = zz[outside_sun_mask & inside_earth_orbit_mask]

v_cube_list = [] #List of cubes - populate with all found model options on the coords
d_cube_list = []

n_cubes = 70

original_densities = []
original_velocities = []
original_speeds = []

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
    # velocity = velocity / 10
    speed = np.sqrt(velocity[...,0]**2+velocity[...,1]**2+velocity[...,2]**2)
    original_densities.append(density)
    original_velocities.append(velocity)
    original_speeds.append(speed)
original_densities = np.asarray(original_densities)
original_velocities = np.asarray(original_velocities)
original_speeds = np.asarray(original_speeds)

global_rho_min = original_densities.min()
global_rho_max = original_densities.max()
global_speed_min = original_speeds.min()
global_speed_max = original_speeds.max()
# original_densities has shape [n_cubes, num_points, num_points, num_points, 1]
# original_velocities has shape [n_cubes, num_points, num_points, num_points, 3]


for checkpoint_path in chk_additionals:
        additional_loader = SuNeRFLoader(checkpoint_path, resolution=512)
        densities = [] #1d - Generates Density at each point in each cube
        velocities = [] #3d - 3 Velocity at each point in each cube     
        for i, timei in tqdm(enumerate(pd.date_range(additional_loader.start_time, additional_loader.end_time, n_cubes)), total=n_cubes):    
                time = normalize_datetime(timei)
                t = np.ones_like(x_filtered) * time
                query_points_npy = np.stack([x_filtered, y_filtered, z_filtered, t], -1).astype(np.float32)
                # (256, 258, 4)

                query_points = torch.from_numpy(query_points_npy)
                enc_query_points = additional_loader.encoding_fn(query_points.view(-1, 4))
                raw = additional_loader.fine_model(enc_query_points)
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
                # velocity = velocity / 10
                densities.append(density)
                velocities.append(velocity)
        v_cube_list.append(velocities)
        d_cube_list.append(densities)
v_cube_list = np.asarray(v_cube_list)
d_cube_list = np.asarray(d_cube_list)
#print("Density shape: {} - Velocity shape: {}".format(d_cube_list.shape, v_cube_list.shape))
density_error = np.zeros((d_cube_list.shape[1], d_cube_list.shape[2]))
# Causes overflow. Recalculate via maxima rescaling (any constant rescaling would do the trick)
# We have std(X) = sqrt(Var(x)) and Var(ax+b) = a^2Var(x) -> b = 0, a = 1/max(x) -> std(x/max(x)) = sqrt(var(x/max(x))) = sqrt(var(x))/max(x) = sigma(X)/max(x) -> std(X) = max(x)*std(x/max(x))
# the constant chosen here doesnt really matter - works with any of them. So, choose something of order x.
for i in range(d_cube_list.shape[2]):
        c = d_cube_list[:,:,i].max() # Local rescaling
        proposed_value = c*(d_cube_list[:,:,i]/c).std(axis = 0)
        # Infinities can happen in the masked region apparently - watch out.
        density_error[:, i] =  proposed_value if not (np.isnan(proposed_value).any() or np.isinf(proposed_value).any()) else np.NaN
        
velocity_error = np.std(v_cube_list, axis = 0)
speed_error = np.sqrt(velocity_error[...,0]**2+velocity_error[...,1]**2+velocity_error[...,2]**2)# Speed associated with velocity_error
# densities has shape [num_models, n_cubes, num_points_encoded 1]
# velocities has shape [num_models, n_cubes, num_points_encoded 3]
# Deviation has be calculated on a per point, basis.
# Create Masks by selecting applicable error ranges

perc_density_offset = 0.1 #Allow for a deviation of 10% from the expected portion
perc_speed_offset = 0.1 #Again use speed for restricting velocities - Allow for a deviation of 10% from the selected voxel

density_offset = np.abs((density_error/(epsilon + original_densities)) -1)
speed_offset = np.abs((speed_error/(epsilon+original_speeds)) - 1)

density_mask = density_offset < perc_density_offset
speed_mask = speed_offset < perc_speed_offset

combined_mask = density_mask & speed_mask


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

density_filenames = []
velocity_filenames = []
masked_density_filenames = []
masked_velocity_filenames = []
for i, (rho, v_abs, error_rho, error_v) in enumerate(zip(original_densities, original_speeds, density_error, speed_error)):
        x_fil_dens = x_filtered[density_mask[i]]
        y_fil_dens = y_filtered[density_mask[i]]
        z_fil_dens = z_filtered[density_mask[i]]

        x_fil_v = x_filtered[speed_mask[i]]
        y_fil_v = y_filtered[speed_mask[i]]
        z_fil_v = z_filtered[speed_mask[i]]

        x_fil_dens_v = x_filtered[combined_mask[i]]
        y_fil_dens_v = y_filtered[combined_mask[i]]
        z_fil_dens_v = z_filtered[combined_mask[i]]
        density_filename = showcase_datacube_with_errors(masked_cube = rho[density_mask[i]], errors_cube = error_rho[density_mask[i]], global_min = global_rho_min, global_max = global_rho_max, tag = "density", idx = i, x_fil = x_fil_dens, y_fil = y_fil_dens, z_fil = z_fil_dens,  alpha_expon = 3, norm = "log")
        velocity_filename = showcase_datacube_with_errors(masked_cube = v_abs[speed_mask[i]], errors_cube = error_v[speed_mask[i]], global_min = global_speed_min, global_max = global_speed_max, tag = "speed", idx = i, x_fil = x_fil_v, y_fil = y_fil_v, z_fil = z_fil_v,  alpha_expon = 3, norm = "linear")
        density_filename_combined = showcase_datacube_with_errors(masked_cube = rho[combined_mask[i]], errors_cube = error_rho[combined_mask[i]], global_min = global_rho_min, global_max = global_rho_max, tag = "density", idx = i, x_fil = x_fil_dens_v, y_fil = y_fil_dens_v, z_fil = z_fil_dens_v,  alpha_expon = 3, norm = "log", fname_subtag = "combined")
        velocity_filename_combined = showcase_datacube_with_errors(masked_cube = v_abs[combined_mask[i]], errors_cube = error_v[combined_mask[i]], global_min = global_speed_min, global_max = global_speed_max, tag = "speed", idx = i, x_fil = x_fil_dens_v, y_fil = y_fil_dens_v, z_fil = z_fil_dens_v,  alpha_expon = 3, norm = "linear", fname_subtag = "combined")
        density_filenames.append(density_filename)
        velocity_filenames.append(velocity_filename)
        masked_density_filenames.append(density_filename_combined)
        masked_velocity_filenames.append(velocity_filename_combined)
frame_duration = 0.5 #2fps
print("Creating Animations")
if len(density_filenames):
    with imageio.get_writer(os.path.join(video_path,'density.gif'), mode='I', duration=frame_duration) as writer:
        for filename in density_filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)
if len(velocity_filenames):
    with imageio.get_writer(os.path.join(video_path,'velocity.gif'), mode='I', duration=frame_duration) as writer:
        for filename in velocity_filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)
if len(masked_velocity_filenames):
    with imageio.get_writer(os.path.join(video_path,'masked_velocity.gif'), mode='I', duration=frame_duration) as writer:
        for filename in masked_velocity_filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)
if len(masked_density_filenames):
    with imageio.get_writer(os.path.join(video_path,'masked_density.gif'), mode='I', duration=frame_duration) as writer:
        for filename in masked_density_filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)

