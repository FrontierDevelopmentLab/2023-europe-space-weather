import datetime
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from tqdm import tqdm
from sunpy.visualization.colormaps import cm
import pandas as pd

from sunerf.data.utils import sdo_cmaps
from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime

from sunpy.map import Map

from datetime import timedelta

from IPython.display import HTML
from base64 import b64encode
import cv2 # pip install opencv-python

import glob

import torch

chk_path = '/mnt/ground-data/training/HAO_v2/save_state.snf'
video_path = '/mnt/ground-data/training/plotting/video_cme'

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
cmap = cm.soholasco2.copy()
cmap.set_bad(color='black') # green

time = loader.start_time + timedelta(days=0.7)

au = (1 * u.AU).to(u.solRad).value
n_points = 40

# start from vigil - let errupt
points_1 = zip(np.ones(n_points) * 0,
               np.ones(n_points) * -190,
               pd.date_range(loader.start_time, time, n_points),
               np.ones(n_points) * 1 * au)

# pan around to Earth
points_2 = zip(np.ones(n_points) * 0,
               np.linspace(-190, -130, n_points),
               [time] * n_points,
               np.ones(n_points) * 1 * au)

# get hit by the CME
points_3 = zip(np.ones(n_points) * 0,
               np.ones(n_points) * -130,
               pd.date_range(time, loader.end_time, n_points),
               np.ones(n_points) * 1 * au)


# combine coordinates
points = [] #list(points_1) + list(points_2) + list(points_3)
strides = 1

# manually add mask base on target example image
mask = np.isnan(Map('/mnt/ground-data/prep_HAO/dcmer_020W_bang_0000_pB_stepnum_005.fits').data)
mask = mask[::strides, ::strides]

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat, lon, time, distance=d, batch_size=4096 * torch.cuda.device_count(), strides=strides)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    outputs['pixel_B'][mask] = np.nan
    outputs['density_map'][mask] = np.nan
    axs[0].imshow(outputs['pixel_B'][..., 0], cmap=cmap, vmin=0, vmax=1, origin='lower')
    axs[1].imshow(outputs['pixel_B'][..., 1], cmap=cmap, vmin=0, vmax=1, origin='lower')
    axs[2].imshow(outputs['density_map'], cmap='viridis', origin='lower')
    axs[0].set_axis_off(), axs[1].set_axis_off(), axs[2].set_axis_off()
    axs[0].set_title("Total Brightness")
    axs[1].set_title("Polarised Brightness")
    axs[2].set_title("Density")
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(video_path, '%03d.jpg' % i))
    plt.close(fig)


video_name = os.path.join(video_path, 'video.mp4')

images = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
frame = cv2.imread(images[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 10, (width,height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()

#sys.exit()


