import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tvtk.api import tvtk, write_data

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime

base_path = '/mnt/training/OBS_v6'
chk_path = os.path.join(base_path, 'save_state.snf')
save_path = os.path.join(base_path, 'vtk_64')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init loader
resolution = 64
loader = SuNeRFLoader(chk_path)

n_time_points = 100
batch_size = 4096 * 4 * torch.cuda.device_count()
os.makedirs(save_path, exist_ok=True)


def save_vtk(vec, path, name, scalar=None, scalar_name='scalar', sr_per_pix=1):
    """Save numpy array as VTK file

    :param vec: numpy array of the vector field (x, y, z, c)
    :param path: path to the target VTK file
    :param name: label of the vector field (e.g., B)
    :param Mm_per_pix: pixel size in Mm. 360e-3 for original HMI resolution. (default bin2 pixel scale)
    """
    # Unpack
    dim = vec.shape[:-1]
    # Generate the grid
    pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.int64) * sr_per_pix
    # reorder the points and vectors in agreement with VTK
    # requirement of x first, y next and z last.
    pts = pts.transpose(2, 1, 0, 3)
    pts = pts.reshape((-1, 3))
    vectors = vec.transpose(2, 1, 0, 3)
    vectors = vectors.reshape((-1, 3))

    sg = tvtk.StructuredGrid(dimensions=dim, points=pts)
    sg.point_data.vectors = vectors
    sg.point_data.vectors.name = name
    if scalar is not None:
        scalars = scalar.transpose(2, 1, 0)
        scalars = scalars.reshape((-1))
        sg.point_data.add_array(scalars)
        sg.point_data.get_array(1).name = scalar_name
        sg.point_data.update()

    write_data(sg, path)


with torch.no_grad():
    for timei in tqdm(pd.date_range(loader.start_time, loader.end_time, n_time_points), total=n_time_points):
        # DENSITY SLICE
        time = normalize_datetime(timei)

        query_points_npy = np.stack(np.meshgrid(
            np.linspace(-15, 15, resolution, dtype=np.float32),
            np.linspace(-15, 15, resolution, dtype=np.float32),
            np.linspace(-15, 15, resolution, dtype=np.float32),
            np.ones((1,), dtype=np.float32) * time, indexing='ij'), -1)

        radius = np.sqrt(np.sum(query_points_npy[:, :, :, 0, :3] ** 2, axis=-1))
        mask = (radius < 4) | (radius > 15)

        r2_mask = radius ** 2 #((radius - 20) / (60 - 20)) ** 2
        # r2_mask = np.clip(r2_mask, 0, 1)

        query_points = torch.from_numpy(query_points_npy)

        # Prepare points --> encoding.
        query_points = query_points.view(-1, 4)

        print('load cube')
        density, velocity = [], []
        for i in range(np.ceil(query_points.shape[0] / batch_size).astype(int)):
            batch = loader.encoding_fn(query_points[i * batch_size:(i + 1) * batch_size])
            raw = loader.fine_model(batch.to(device))
            density += [raw[..., 0].cpu().detach()]
            velocity += [raw[..., 1:].cpu().detach()]

        # stack results
        density = torch.cat(density, dim=0)
        velocity = torch.cat(velocity, dim=0)
        # reshape
        density = density.view(query_points_npy.shape[:3]).cpu().detach().numpy()
        velocity = velocity.view(query_points_npy.shape[:3] + velocity.shape[-1:]).cpu().detach().numpy()
        #
        # scale density
        density *= r2_mask
        #
        velocity = velocity  # * density[..., None] / 1e27 # scale to mass flux
        # apply mask
        density[mask] = 0  # np.nan
        velocity[mask] = 0  # np.nan

        print('save vtk')
        vtk_filename = os.path.join(save_path, f"data_cube_{timei.isoformat('T', timespec='minutes')}.vtk")
        save_vtk(velocity, vtk_filename, "v", density, "density", sr_per_pix= 200 / resolution)
