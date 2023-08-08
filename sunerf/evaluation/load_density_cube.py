import os

import numpy as np
import pandas as pd
import torch
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime

from tvtk.api import tvtk, write_data

base_path = '/mnt/training/HAO_pinn_cr_2view_a26978f_heliographic_reformat'
chk_path = os.path.join(base_path, 'save_state.snf')
video_path_dens = os.path.join(base_path, 'video_density')

def save_vtk(vec, path, name, scalar=None, scalar_name='scalar', Mm_per_pix=1):
    """Save numpy array as VTK file

    :param vec: numpy array of the vector field (x, y, z, c)
    :param path: path to the target VTK file
    :param name: label of the vector field (e.g., B)
    :param Mm_per_pix: pixel size in Mm. 360e-3 for original HMI resolution. (default bin2 pixel scale)
    """
    # Unpack
    dim = vec.shape[:-1]
    # Generate the grid
    pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.int64) * Mm_per_pix
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

# init loader
loader = SuNeRFLoader(chk_path, resolution=512)
n_points = 40
os.makedirs(video_path_dens, exist_ok=True)

densities = []
for i, timei in tqdm(enumerate(pd.date_range(loader.start_time, loader.end_time, n_points)), total=n_points):
    # TEST FOR KNOWN LOCATION
    lati = 0
    loni = 272.686
    di = 214.61000061  # (* u.m).to(u.solRad).value

    # DENSITY SLICE
    time = normalize_datetime(timei)

    query_points_npy = np.stack(np.mgrid[-100:100:2, -100:100:2, -100:100:2, 1:2], -1).astype(np.float32)

    mask = np.sqrt(np.sum(query_points_npy[:, :, 0, 0, :3] ** 2, axis=-1)) < 21

    query_points = torch.from_numpy(query_points_npy)
    query_points[..., -1] = time

    # Prepare points --> encoding.
    enc_query_points = loader.encoding_fn(query_points.view(-1, 4))

    raw = loader.fine_model(enc_query_points)
    density = raw[..., 0]
    velocity = raw[..., 1:]

    density = density.view(query_points_npy.shape[:3]).cpu().detach().numpy()
    velocity = velocity.view(query_points_npy.shape[:3] + velocity.shape[-1:]).cpu().detach().numpy()
    velocity = velocity / 10 #* density[..., None] / 1e27 # scale to mass flux
    # apply mask
    density[mask] = np.nan
    velocity[mask] = np.nan
    # print(density.max(), density.min())
    densities += [density]

    vtk_filename = os.path.join(base_path,"data_cube_{}.vtk".format(i))
    save_vtk(velocity, vtk_filename, "v", density, "density" )

