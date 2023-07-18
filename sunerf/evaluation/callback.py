from typing import Optional

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize

from sunerf.data.utils import sdo_img_norm
from astropy import units as u

from sunerf.utilities.data_loader import unnormalize_datetime


def plot_samples(channel_map, channel_map_coarse, height_map, absorption_map, testimg, z_vals_stratified,
                 z_vals_hierach, distance, cmap):
    # Log example images on wandb
    # # Plot example outputs

    fig, ax = plt.subplots(1, 6, figsize=(30, 4))

    ax[0].imshow(testimg[..., 0], cmap=cmap, norm=sdo_img_norm)
    ax[0].set_title(f'Target')
    ax[1].imshow(channel_map[..., 0], cmap=cmap, norm=sdo_img_norm)
    ax[1].set_title(f'Prediction')
    ax[2].imshow(channel_map_coarse[..., 0], cmap=cmap, norm=sdo_img_norm)
    ax[2].set_title(f'Coarse')
    ax[3].imshow(height_map, cmap='plasma') #TODO , vmin=1, vmax=1.3)
    ax[3].set_title(f'Emission Height')
    ax[4].imshow(absorption_map, cmap='viridis' )#TODO, vmin=0)
    ax[4].set_title(f'Absorption')

    # select index
    y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4 # select point in first quadrant
    plot_ray_sampling(z_vals_stratified[y, x] - distance, z_vals_hierach[y, x] - distance, ax[-1])

    wandb.log({"Comparison": fig})
    plt.close('all')


def log_overview(images, poses, times, cmap):
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    colors = cm.get_cmap('viridis')(Normalize()(times))
    # fix arrow heads (2) + shaft color (2) --> 3 color elements
    cs = colors.tolist()
    for c in colors:
        cs.append(c)
        cs.append(c)

    iter_list = list(enumerate(images))
    step = max(1, len(iter_list) // 50)
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
        ax.imshow(img[..., 0], norm=sdo_img_norm,  cmap=cmap)
        ax.set_axis_off()
        ax.set_title('Time: %s' % unnormalize_datetime(times[i]).isoformat(' '))

        ax = plt.subplot(133)
        # plot corresponding image
        ax.imshow(img[..., 1], norm=sdo_img_norm, cmap=cmap)
        ax.set_axis_off()
        ax.set_title('Time: %s' % unnormalize_datetime(times[i]).isoformat(' '))

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
