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
cmap.set_bad(color='white') # green

time = loader.start_time + timedelta(days=1)

au = (1 * u.AU).to(u.solRad).value
n_points = 10

# over lons
points_1 = zip(np.ones(n_points) * 0,
               np.linspace(0, 360, n_points),
               [time] * n_points,
               np.ones(n_points) * 1 * au)

# over lats
points_2 = zip(np.linspace(0, 360, n_points),
               np.ones(n_points) * 0,
               [time] * n_points,
               np.ones(n_points) * 1 * au)

# zoom (distance of observe) or watch forward in time from same point
points_3 = zip(np.ones(n_points) * 0,
               np.linspace(0, 60, n_points),
               pd.date_range(time, loader.start_time, n_points),
               np.ones(n_points) * 1 * au)

points_4 = zip(np.ones(n_points) * 0,
               np.linspace(60, -60, n_points),
               pd.date_range(loader.start_time, loader.end_time, n_points),
               np.ones(n_points) * 1 * au)

# combine coordinates
points = list(points_1) + list(points_2) + list(points_3) + list(points_4)

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat, lon, time, distance=d, batch_size=4096 * 1, strides=1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
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



# TEST FOR KNOWN LOCATION
lati = 0
loni = 272.686
timei = datetime.datetime(2010, 4, 13, 19, 40)
di = 214.61000061 # (* u.m).to(u.solRad).value
outputs = loader.load_observer_image(lati, loni, timei, distance=di, batch_size=4096 * 1, strides=1)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(outputs['pixel_B'][..., 0], cmap=cmap, vmin=0, vmax=1, origin='lower')
axs[1].imshow(outputs['pixel_B'][..., 1], cmap=cmap, vmin=0, vmax=1, origin='lower')
axs[2].imshow(outputs['density_map'], cmap='viridis', origin='lower')
axs[0].set_axis_off(), axs[1].set_axis_off(), axs[2].set_axis_off()
axs[0].set_title("Total Brightness")
axs[1].set_title("Polarised Brightness")
axs[2].set_title("Density")
plt.tight_layout(pad=0)
fig.savefig(os.path.join(video_path, 'test_str1.jpg'))
plt.close(fig)



time = normalize_datetime(timei)

query_points_npy = np.stack(np.mgrid[-100:100, -100:100, 0:1, 1:2], -1).astype(np.float32)

query_points = torch.from_numpy(query_points_npy)
query_points[..., -1] = time

# Prepare points --> encoding.
enc_query_points = loader.encoding_fn(query_points.view(-1, 4))

# Coarse model pass.
raw = loader.coarse_model(enc_query_points)
density = 10 ** raw

density = density.view(query_points_npy.shape[:2])

fig = plt.figure()
plt.imshow(density.cpu().detach().numpy(), norm='log')
fig.savefig(os.path.join(video_path, 'dens_slice.jpg'))
plt.close(fig)