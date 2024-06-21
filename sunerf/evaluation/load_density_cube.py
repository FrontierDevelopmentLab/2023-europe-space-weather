import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.evaluation.vtk import save_vtk

parser = argparse.ArgumentParser(description='Evaluate density cube')
parser.add_argument('--ckpt_path', type=str, help='path to the SuNeRF checkpoint')
parser.add_argument('--result_path', type=str, help='path to the result directory')
args = parser.parse_args()

chk_path = args.ckpt_path
result_path = args.result_path
os.makedirs(result_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
n_points = 100
resolution = 128
max_radius = 100 / loader.Rs_per_ds
min_radius = (0.1 * u.AU).to_value(u.solRad) / loader.Rs_per_ds

for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_points)), total=n_points):
    # DENSITY SLICE
    time = loader.normalize_datetime(timei)

    query_points_npy = np.stack(np.meshgrid(
        np.linspace(-max_radius, max_radius, resolution),
        np.linspace(-max_radius, max_radius, resolution),
        np.linspace(-max_radius, max_radius, resolution),
        [time]), -1).astype(np.float32)
    query_points_npy = query_points_npy[:, :, :, 0, :]

    coords = query_points_npy[..., :3]

    radius = np.sqrt(np.sum(coords ** 2, axis=-1))
    mask = (radius < min_radius) | \
           (radius > max_radius)

    sub_coords = query_points_npy[~mask]
    model_out = loader.load_coords(sub_coords)

    density = np.zeros(query_points_npy.shape[:-1], dtype=np.float32)
    density[~mask] = model_out['density']
    velocity = np.zeros(query_points_npy.shape[:-1] + (3,), dtype=np.float32)
    velocity[~mask] = model_out['velocity']


    vtk_filename = os.path.join(result_path, f"data_cube_{i:03d}.vtk")
    save_vtk(vtk_filename, coords=coords, vectors={'velocity': velocity}, scalars={'density': density})
