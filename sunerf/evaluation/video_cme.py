import argparse
import glob
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader

parser = argparse.ArgumentParser(description='Evaluate density cube')
parser.add_argument('--ckpt_path', type=str, help='path to the SuNeRF checkpoint')
parser.add_argument('--result_path', type=str, help='path to the result directory')
args = parser.parse_args()

chk_path = args.ckpt_path
result_path = args.result_path
os.makedirs(result_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
cmap = cm.soholasco2.copy()
# cmap.set_bad(color='white')  # green

time = loader.start_time + timedelta(days=0.5)

au = (1 * u.AU).to(u.solRad).value
n_points = 60

# start from vigil - let errupt
points_1 = zip(np.ones(n_points) * 0,
               np.ones(n_points) * -160,
               pd.date_range(loader.start_time, time, n_points),
               np.ones(n_points) * 1 * au)

# pan around to Earth
points_2 = zip(np.ones(n_points) * 0,
               np.linspace(-160, -250, n_points),
               [time] * n_points,
               np.ones(n_points) * 1 * au)

# get hit by the CME
points_3 = zip(np.ones(n_points) * 0,
               np.ones(n_points) * (-250),
               pd.date_range(time, loader.end_time, n_points),
               np.ones(n_points) * 1 * au)

# combine coordinates
points = list(points_1) + list(points_2) + list(points_3)
strides = 1
norm = LogNorm()
for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat, lon, time, distance=d, batch_size=4096 * torch.cuda.device_count(),
                                         strides=strides)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(outputs['tB'], cmap=cmap, origin='lower', norm=norm)
    axs[1].imshow(outputs['pB'], cmap=cmap, origin='lower', norm=norm)
    axs[2].imshow(outputs['density_map'], cmap='inferno', origin='lower')
    axs[0].set_axis_off(), axs[1].set_axis_off(), axs[2].set_axis_off()
    axs[0].set_title("Total Brightness")
    axs[1].set_title("Polarised Brightness")
    axs[2].set_title("Integrated Density")
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(result_path, '%03d.png' % i), transparent=True)
    plt.close(fig)

