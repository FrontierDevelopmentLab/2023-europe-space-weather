import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime
import cv2

from tvtk.api import tvtk, write_data

from mpl_toolkits.mplot3d import Axes3D

import imageio #gifs
import logging
import sys

#import vtk
#import tvtk

#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # Output to console.
from scipy.stats import multivariate_normal

base_path = '/mnt/training/HAO_pinn_cr_2view_a26978f_heliographic_reformat'

chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_cube')


#parameter for filtering points that belong to a CME
mask_mode = True

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
os.makedirs(video_path_dens, exist_ok=True)
n_cubes = 70 # How many cubes to generate

# Points in R_solar
num_points = 16

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

percentile = 95 # Use this percentile for masking
percentile = np.clip(percentile, 0, 100)
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
    mag = np.sqrt(velocity[...,0]**2+velocity[...,1]**2+velocity[...,2]**2)

    densities.append(density)
    velocities.append(velocity)
    speeds.append(mag)


global_max_v = np.asarray(speeds).max()
global_min_v = np.asarray(speeds).min()
global_max_rho = np.asarray(densities).max()
global_min_rho = np.asarray(densities).min()

mean_density = np.asarray(densities).mean()
mean_velocity = np.asarray(speeds).mean()

perc_dens = np.percentile(np.asarray(densities),percentile)
perc_speed = np.percentile(np.asarray(speeds),percentile)
print("Mean Density: {} g per cm^3 \n {}% Percentile:{} g per cm^3 \n Max Density: {} g per cm^3 \n Min Density: {:.3f} g per cm^3.".format(mean_density,percentile, perc_dens,global_max_rho, global_min_rho))
print("Mean Velocity: {:.3f} Solar Radii per 2 days \n {}% Percentile:{:.3f} Solar Radii per 2 days \n Max Velocity: {:.3f} Solar Radii per 2 days \n Min Velocity: {:.3f} Solar Radii per 2 days.".format(mean_velocity,percentile,perc_speed,global_max_v, global_min_v))
# Choose thresholds on some percentile
density_threshold = perc_dens
velocity_threshold = perc_speed #Most CMEs appear to be moving with a velocity of 3
print("Thresholding density at {} grams per cm^3, velocity at {:.3f} Solar Radii per 2 days.".format(density_threshold, velocity_threshold))

def compute_alpha(alpha_mode, cube_norm, alpha_expon,distance_from_sun):
    if alpha_mode==0:
        alpha = cube_norm**alpha_expon 
    elif alpha_mode==1:
        sig_arg = cube_norm**alpha_expon + distance_from_sun
        alpha = np.tanh(sig_arg)

def plot_datacube_directly(cube, global_min,global_max, tag, idx, x_fil, y_fil,z_fil, alpha_expon, norm = "linear", fname_subtag = None, alpha_mode=1):
    cube_norm = (cube - global_min)/(global_max - global_min)
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    distance_from_sun = np.sqrt(x_fil**2 + y_fil**2 + z_fil**2)/250
    alpha = compute_alpha(alpha_mode, cube_norm, alpha_expon,distance_from_sun)
    if(len(x_fil) and len(y_fil) and len(z_fil)):
        ax.scatter(x_fil, y_fil, z_fil, c=cube, marker='.',norm=norm , vmin=global_min, vmax=global_max, alpha = alpha) #
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
    x_earth = np.outer(np.cos(u), np.sin(v))*7# * radius_earth
    y_earth = np.outer(np.sin(u), np.sin(v))*7# * radius_earth
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))*7# * radius_earth

    # Earth position defined on x axis, 1 AU = 215.032 Solar Radii
    r_earth = 215.032
    center_earth = (-r_earth, 0, 0)
    # If center_earth = (-r_earth, 0, 0) then on a circle, this is at pi radians. If L5 is trailing, then L5 is at (r_earth*cos(2/3*np.pi), r_earth*sin(2/3*np.pi), 0)
    center_l5 = (r_earth*np.cos(2/3*np.pi), r_earth*np.sin(2/3*np.pi), 0)

    x_l5 = x_earth + center_l5[0]
    y_l5 = x_earth + center_l5[1]
    z_l5 = x_earth + center_l5[2]

    x_earth += center_earth[0]
    y_earth += center_earth[1]
    z_earth += center_earth[2]

    # Plot the Earth
    ax.plot_surface(x_earth, y_earth, z_earth, color='cyan', alpha=1)
    # Plot L5
    ax.plot_surface(x_l5, y_l5, z_l5, color = "red", alpha = 1)


    # Plot earth orbit
    x_orbit = r_earth * np.cos(u)
    y_orbit = r_earth * np.sin(u)
    z_orbit = np.zeros_like(u)

    ax.scatter(x_orbit, y_orbit, z_orbit, c='gray', marker='.', alpha = 0.1)

    filename = os.path.join(video_path_dens, f'{tag}_cube_{idx:03d}.jpg')

    if fname_subtag is not None:
        filename = os.path.join(video_path_dens, f'{tag}_{fname_subtag}_cube_{idx:03d}.jpg')
    fig.savefig(filename, dpi=100)
    return filename
    

def plot_datacube(cube,global_min:float, global_max:float, tag:str, idx:int, plot_threshold:float,  alpha_expon:float = 1.5, norm = "linear", alpha_mode=1):
    '''
        Function plotting the datacube, the Sun, Earth and L5, alongside Earths orbit
    '''
    cube_norm = (cube - global_min)/(global_max - global_min)
    plt.close("all")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.add_subplot(121, projection='3d')  # Main 3D scatter plot
    #ax_hist_x = fig.add_subplot(322)  # Histogram for x-axis
    #ax_hist_y = fig.add_subplot(324)  # Histogram for y-axis
    #ax_hist_z = fig.add_subplot(326)  # Histogram for z-axis

    # Only plot values above a threshold
    mask = cube > plot_threshold

    x_filtered_again = x_filtered[mask]
    y_filtered_again = y_filtered[mask]
    z_filtered_again = z_filtered[mask]
    
    cube = cube[mask]
    cube_norm = cube_norm[mask]

    distance_from_sun = np.sqrt(x_filtered_again**2 + y_filtered_again**2 + z_filtered_again**2)/250
    alpha = compute_alpha(alpha_mode, cube_norm, alpha_expon,distance_from_sun)
    ax.scatter(x_filtered_again, y_filtered_again, z_filtered_again, c=cube, marker='.',norm=norm , vmin=global_min, vmax=global_max, alpha = alpha) #
    ax.set_xlim(-250,250)
    ax.set_ylim(-250,250)
    ax.set_zlim(-250,250)
    ax.set_xlabel('X[ Solar Radii ]')
    ax.set_ylabel('Y[ Solar Radii ]')
    ax.set_zlabel('Z[ Solar Radii ]')
    ax.set_title('3D Scatter Plot of points based on {} at timestep {}'.format(tag, idx))
    if norm == "log":
        ticks = np.linspace(np.log(global_min), np.log(global_max), 10, endpoint = True)
    else:
        ticks= np.linspace(global_min,global_max,10, endpoint = True)
    cbar = plt.colorbar(ax.collections[0], ax=ax, ticks = ticks)
    cbar.set_label('{}'.format(tag))
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
    x_earth = np.outer(np.cos(u), np.sin(v))*7# * radius_earth
    y_earth = np.outer(np.sin(u), np.sin(v))*7# * radius_earth
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))*7# * radius_earth

    # Earth position defined on x axis, 1 AU = 215.032 Solar Radii
    r_earth = 215.032
    center_earth = (-r_earth, 0, 0)
    # If center_earth = (-r_earth, 0, 0) then on a circle, this is at pi radians. If L5 is trailing, then L5 is at (r_earth*cos(2/3*np.pi), r_earth*sin(2/3*np.pi), 0)
    center_l5 = (r_earth*np.cos(2/3*np.pi), r_earth*np.sin(2/3*np.pi), 0)
    x_l5 = x_earth + center_l5[0]
    y_l5 = x_earth + center_l5[1]
    z_l5 = x_earth + center_l5[2]

    x_earth += center_earth[0]
    y_earth += center_earth[1]
    z_earth += center_earth[2]

    # Plot the Earth
    ax.plot_surface(x_earth, y_earth, z_earth, color='cyan', alpha=1)
    # Plot L5
    
    ax.plot_surface(x_l5, y_l5, z_l5, color = "red", alpha = 1)


    # Plot earth orbit
    x_orbit = r_earth * np.cos(u)
    y_orbit = r_earth * np.sin(u)
    z_orbit = np.zeros_like(u)

    ax.scatter(x_orbit, y_orbit, z_orbit, c='gray', marker='.', alpha = 0.1)

    # Histograms along each axis
    #ax_hist_x.hist(x_filtered_again, bins=15, color='b', alpha=0.6)
    #ax_hist_y.hist(y_filtered_again, bins=15, color='g', alpha=0.6)
    #ax_hist_z.hist(z_filtered_again, bins=15, color='r', alpha=0.6)

    # Set titles for histograms
    #ax_hist_x.set_title('Histogram along X axis')
    #ax_hist_y.set_title('Histogram along Y axis')
    #ax_hist_z.set_title('Histogram along Z axis')

    filename = os.path.join(video_path_dens, f'{tag}_cube_{idx:03d}.jpg')
    fig.savefig(filename, dpi=100)
    return filename
    
    
density_filenames = []
velocity_filenames = []
masked_density_filenames = []
masked_velocity_filenames = []

masked_density = []
masked_velocity = []

last_mask = None
mean_velocity = []

for i, (rho, v, abs_v) in enumerate(zip(densities, velocities, speeds)):
    density_filename = plot_datacube(rho,global_min_rho, global_max_rho, tag = "density", idx = i,plot_threshold = density_threshold,  alpha_expon = 3, norm = "log")
    velocity_filename = plot_datacube(abs_v,global_min_v, global_max_v, tag = "velocity", idx = i,plot_threshold = velocity_threshold,  alpha_expon = 3)
    density_filenames.append(density_filename)
    velocity_filenames.append(velocity_filename)
    if mask_mode:
        density_mask = rho > density_threshold
        velocity_mask = abs_v > velocity_threshold
        mask = density_mask & velocity_mask   #
        if last_mask is not None:
            # Last mask needs to be blurred - we want to remove voxels around the currently active points, as well as the points themselves
            # last_mask exists, therefore we take out the background
            last_mask = ~last_mask 
            #Every spot that has been accepted last time is now disabled
            # every new spot is still possible - achieving recent background subtraction
            mask = mask & last_mask
            
        # Positions that belong only to outliers
        x_filtered_again = x_filtered[mask]
        y_filtered_again = y_filtered[mask]
        z_filtered_again = z_filtered[mask]
        # Values that belong only to outliers
        masked_rho = rho[mask]
        masked_v = abs_v[mask]
        masked_density_fname = plot_datacube_directly(masked_rho,global_min_rho, global_max_rho, tag = "density",x_fil = x_filtered_again, y_fil = y_filtered_again, z_fil = z_filtered_again, idx = i,  alpha_expon = 3, norm = "log", fname_subtag = "masked")
        masked_velocity_fname = plot_datacube_directly(masked_v,global_min_v, global_max_v, tag = "velocity", x_fil = x_filtered_again, y_fil = y_filtered_again, z_fil = z_filtered_again, idx = i,  alpha_expon = 3, fname_subtag = "masked")
        masked_density_filenames.append(masked_density_fname)
        masked_velocity_filenames.append(masked_velocity_fname)

        vmean = [0,0,0]
        if len(v[mask]):
            vmean = v[mask].mean(axis = 0)
        mean_velocity.append(vmean)
        masked_density.append(masked_rho)
        masked_velocity.append(masked_v)
        if last_mask is None:
            last_mask = mask #set up last mask as the first possible mask - otherwise might flicker


frame_duration = 0.5 #2fps
if len(density_filenames):
    with imageio.get_writer(os.path.join(video_path_dens,'density.gif'), mode='I', duration=frame_duration) as writer:
        for filename in density_filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)
if len(velocity_filenames):
    with imageio.get_writer(os.path.join(video_path_dens,'velocity.gif'), mode='I', duration=frame_duration) as writer:
        for filename in velocity_filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)

if mask_mode:
    if len(masked_velocity_filenames):
        with imageio.get_writer(os.path.join(video_path_dens,'masked_velocity.gif'), mode='I', duration=frame_duration) as writer:
            for filename in masked_velocity_filenames:
                image = imageio.v3.imread(filename)
                writer.append_data(image)
    if len(masked_density_filenames):
        with imageio.get_writer(os.path.join(video_path_dens,'masked_density.gif'), mode='I', duration=frame_duration) as writer:
            for filename in masked_density_filenames:
                image = imageio.v3.imread(filename)
                writer.append_data(image)


def calculate_mean_velocity_and_density(density_cube:np.ndarray, velocity_cube:np.ndarray, speed_cube:np.ndarray, percentile:float, speeds:np.array = None, densities:np.array = None, last_mask = None) -> dict:
    '''
        Function: Calculate mean velocity:
            Given the density and velocity cubes, calculates the mask needed to extract the CME.
            Using the mask, calculates 
                the mean velocity (speed and direction) of the CME (in Solar Radii per 2 days), 
            alongside
                the mean density (in N_e per cm^3),
            
        Input:
            density_cube: np.ndarray: Density cube used extracted from fine model in the SuNeRF
            velocity_cube: np.ndarray: Velocity cube from the same source
            speed_cube: np.ndarray: Generated from velocity cube, vector norm of the velocities
            percentile: Float: percentile used to set the threshold of speeds and densities
            speeds: np.array : optional: default None: Array of speeds to use for calculating the thresholds. 
                                         If None, calculates this array from passed velocity_cube
            densities: np.array: optional: default None:  Array of densities used for calculating a threshold.
                                         If None, calculates array from density_cube
            last_mask: np.array: optional: default None: Mask used to deactivate pixels in background subtraction
        Output:
            Dictionary with 7 keys:
                "Density":    Density of the CME that has been detected in N_e/cm^3
                "Velocity": Mean Velocity Vector of the CME
                "Speed": |v| - vector norm of velocity
                "Direction": |\hat{v}| - Direction of the mean velocity vector
                "Densities": Collection of densities that has been accepted
                "Velocities": Collection of velocities that have been accepted
                "Mask": mask used to accept voxels (in either cube)
    '''

    percentile = np.clip(percentile,0,99)
    if speeds is None:
        speeds = np.flatten(speed_cube)
    if densities is None:
        densities = np.flatten(densities)
    density_threshold = np.percentile(np.asarray(densities),percentile)
    velocity_threshold = np.percentile(np.asarray(speeds),percentile)
    

    density_mask = densities > density_threshold
    velocity_mask = speeds > velocity_threshold
    mask = density_mask & velocity_mask   #
    if last_mask is not None:
        # Last mask needs to be blurred - we want to remove voxels around the currently active points, as well as the points themselves
        # last_mask exists, therefore we take out the background
        last_mask = ~last_mask 
        #Every spot that has been accepted last time is now disabled
        # every new spot is still possible - achieving recent background subtraction
        mask = mask & last_mask
    
    rhomean = None
    if len(density_cube[mask]):
        rhomean = density_cube[mask].mean()
    vmean = None
    if len(velocity_cube[mask]):
        vmean = velocity_cube[mask].mean(axis = 0)
    speedmean = None
    if len(speed_cube[mask]):
        speedmean = speed_cube[mask].mean()
    direction = vmean/(np.dot(vmean,vmean))

    out_dict = {}
    out_dict["Density"] = rhomean
    out_dict["Velocity"] = vmean
    out_dict["Speed"] = speedmean
    out_dict["Direction"] = direction
    out_dict["Densities"] = density_cube[mask]
    out_dict["Velocities"] = velocity_cube[mask]
    out_dict["Mask"] = mask
    
    return out_dict

def estimate_probability_of_hit(earth_position:np.array, mean_velocity:np.array, densities:np.ndarray, velocities:np.ndarray, positions_x:np.array, positions_y:np.array,positions_z:np.array) -> np.array:
    """
    This function estimates the probability of a detected CME hitting earth.
    Currently, this is a simplified model, working with a kernel density estimate for the distribution of velocity.
    
    It estimates the probability of a CME hitting earth within the next two steps, with the following calculations:
    1. Calculate the direction of earth from the CME mean positions
    2. Calculate the KDE of direction vectors from the velocity array
    3. Use KDE to calculate the probability of the CME moving in the direction of earth - P0
    4. Move CME to next position based on velocity (new position vectors)


    Args:
        earth_position (np.array): [x,y,z] - position vector of earth
        mean_velocity (np.array): [vx,vy,vz] - mean velocity vector of the CME - output of calculate_mean_velocity_and_density
        densities (np.ndarray): Density vector that is designated as CME - useful for classification
        velocities (np.ndarray): Velocities of points designated as CME
        positions_x (np.array): Positions that have been designated as CME
        positions_y (np.array): "
        positions_z (np.array): "

    Returns:
        probability: float: probability of the CME hitting earth
    """
    positions = np.stack([positions_x,positions_y,positions_z], axis = 1)
    connection_vector = positions - earth_position
    #direction vectors
    direction_vectors = np.asarray([v/np.dot(v,v) for v in connection_vector]) 
    #direction vector for mean velocity
    #mean_direction = mean_velocity/np.dot(mean_velocity,mean_velocity)
    

    velocity_directions = np.asarray([v/np.dot(v,v) for v in velocities])
    #Distribution of direction vectors
    std_dx,std_dy,std_dz = velocity_directions.std(axis = 0)
    kernel_density_estimate = multivariate_normal(velocity_directions.mean(axis = 0), velocity_directions.std(axis = 0))
    #Use kernel_density_estimate.cdf about regions.
    #Calculate probability of each direction based on the standard deviation of direction vectors
    probabilities = [kernel_density_estimate.cdf(np.asarray([vx+std_dx, vy+std_dy,vz+std_dz])) - kernel_density_estimate.cdf(np.asarray([vx-std_dx, vy-std_dy,vz-std_dz])) for vx,vy,vz in direction_vectors]
    probabilities = np.asarray(probabilities)
    return probabilities