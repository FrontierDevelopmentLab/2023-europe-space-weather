from typing import Optional

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sunerf.data.utils import sdo_img_norm
from astropy import units as u

from sunerf.utilities.data_loader import unnormalize_datetime

# import sunpy
# import copy

def plot_samples(channel_map, channel_map_coarse, distance_sun, distance_obs, density_map, testimg, z_vals_stratified,
                 z_vals_hierach, distance, cmap):
    # Log example images on wandb
    # # Plot example outputs
    b_norm = LogNorm(vmin=np.nanmin(testimg), vmax=np.nanmax(testimg))

    fig, ax = plt.subplots(4, 3, figsize=(20, 15))

    channel_map = np.copy(channel_map)
    channel_map_coarse = np.copy(channel_map_coarse)
    distance_sun = np.copy(distance_sun)
    distance_obs = np.copy(distance_obs)
    density_map = np.copy(density_map)

    # get mask from target image for occulter
    mask = np.isnan(testimg[...,0])
    channel_map[mask, :] = np.nan
    channel_map_coarse[mask, :] = np.nan
    distance_sun[mask] = np.nan
    distance_obs[mask] = np.nan
    density_map[mask] = np.nan

    # TODO AND THEN MAKE SURE GREEN?
    # problem is with sdo_img_norm
    # either manual norm or plot masked circle over top
    # cmap1 = sunpy.visualization.colormaps.cm.soholasco2.copy()
    # cmap1 = copy.copy(sunpy.visualization.colormaps.cm.soholasco2)
    # cmap1.set_bad(color='green')

    # tB
    ax[0,0].imshow(testimg[..., 0], cmap=cmap, norm=b_norm)
    ax[0,0].set_title(f'Target tB')
    ax[0,1].imshow(channel_map[..., 0], cmap=cmap, norm=b_norm)
    ax[0,1].set_title(f'Prediction tB')
    im = ax[0,2].imshow(channel_map_coarse[..., 0], cmap=cmap, norm=b_norm)
    ax[0,2].set_title(f'Coarse tB')
    divider = make_axes_locatable(ax[0,2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # pB
    ax[1,0].imshow(testimg[..., 1], cmap=cmap, norm=b_norm)
    ax[1,0].set_title(f'Target pB')
    ax[1,1].imshow(channel_map[..., 1], cmap=cmap, norm=b_norm)
    ax[1,1].set_title(f'Prediction pB')
    im = ax[1,2].imshow(channel_map_coarse[..., 1], cmap=cmap, norm=b_norm)
    ax[1,2].set_title(f'Coarse pB')
    divider = make_axes_locatable(ax[1,2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # height and density maps
    ax[2,0].imshow(distance_sun[..., 0], cmap='viridis')
    ax[2,0].set_title(f'Distance from Sun')
    ax[2,1].imshow(distance_obs[..., 0], cmap='viridis')
    ax[2,1].set_title(f'Distance from observer')
    ax[2,2].imshow(density_map[..., 0], cmap='viridis' )
    ax[2,2].set_title(f'Density')

    # plot coarse and fine sampling
    # select index
    y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4 # select point in first quadrant
    plot_ray_sampling(z_vals_stratified[y, x] - distance, z_vals_hierach[y, x] - distance, ax[3,0])

    wandb.log({"Comparison": fig})
    plt.close('all')


def log_overview(images, poses, times, cmap, seconds_per_dt, Rs_per_ds, ref_time):
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    colors = cm.get_cmap('viridis')(Normalize()(times))
    # fix arrow heads (2) + shaft color (2) --> 3 color elements
    cs = colors.tolist()
    for c in colors:
        cs.append(c)
        cs.append(c)

    iter_list = list(enumerate(images))
    step = max(1, len(iter_list) // 10)
    for i, img in iter_list[::step]:
        fig = plt.figure(figsize=(16, 8), dpi=150)
        ax = plt.subplot(131, projection='3d')
        # plot all viewpoints
        _ = ax.quiver(
            origins[..., 0].flatten(),
            origins[..., 1].flatten(),
            origins[..., 2].flatten(),
            dirs[..., 0].flatten(),
            dirs[..., 1].flatten(),
            dirs[..., 2].flatten(), color=cs, length=50, normalize=False, pivot='middle',
            linewidth=2, arrow_length_ratio=0.1)

        # plot current viewpoint
        _ = ax.quiver(
            origins[i:i + 1, ..., 0].flatten(),
            origins[i:i + 1, ..., 1].flatten(),
            origins[i:i + 1, ..., 2].flatten(),
            dirs[i:i + 1, ..., 0].flatten(),
            dirs[i:i + 1, ..., 1].flatten(),
            dirs[i:i + 1, ..., 2].flatten(), length=50, normalize=False, color='red', pivot='middle', linewidth=5,
            arrow_length_ratio=0.2)

        d = (1.2 * u.AU).to(u.solRad).value
        ax.set_xlim(-d, d)
        ax.set_ylim(-d, d)
        ax.set_zlim(-d, d)
        ax.scatter(0, 0, 0, marker='o', color='yellow')

        ax = plt.subplot(132)
        # plot corresponding image
        im = ax.imshow(img[..., 0], norm=LogNorm(),  cmap=cmap)
        ax.set_axis_off()
        ax.set_title('Time: %s' % unnormalize_datetime(times[i], seconds_per_dt, ref_time).isoformat(' '))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = plt.subplot(133)
        # plot corresponding image
        im = ax.imshow(img[..., 1], norm=LogNorm(), cmap=cmap)
        ax.set_axis_off()
        ax.set_title('Time: %s' % unnormalize_datetime(times[i], seconds_per_dt, ref_time).isoformat(' '))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        wandb.log({'Overview': fig}, step=i)
        plt.close(fig)


def plot_ray_sampling(
        z_vals: torch.Tensor,
        z_hierarch: Optional[torch.Tensor] = None,
        ax: Optional[np.ndarray] = None):
    r"""
    Plot stratified and (optional) hierarchical samples.
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o', markersize=4)
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o', markersize=4)
    ax.set_ylim([-1, 2])
    # ax.set_xlim([-1.3, 1.3])
    ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
