import argparse
import multiprocessing
import os
from glob import glob
from itertools import repeat

import numpy as np
from astropy import units as u
from sunpy.map import Map


def _loadMLprepMap(file_path, out_path, resolution):
    """Load and preprocess OBS file.


    Parameters
    ----------
    file_path: path to the FTS file.
    resolution: target resolution in pixels.

    Returns
    -------
    the preprocessed SunPy Map
    """
    # load Map
    s_map = Map(file_path)
    # adjust image size
    s_map = s_map.resample((resolution, resolution) * u.pix)

    # normalize image data
    data = s_map.data
    # tB: -23.63556 -15.273365; pB: -29.108622 -18.05343
    v_min, v_max = -29, -15
    data[data > 0] = (np.log(data[data > 0]) - v_min) / (v_max - v_min)
    data[data < 0] = 0  # remove negative values
    data = np.nan_to_num(data, nan=0)
    data = data.astype(np.float32)

    s_map = Map(data, s_map.meta)
    s_map.save(os.path.join(out_path, os.path.basename(file_path)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--obs_path', type=str,
                   default='/mnt/ground-data/data_fits_stereo_2014_02/*/*.fts',
                   help='search path for maps.')
    p.add_argument('--resolution', type=float,
                   default=1024,
                   help='target image scale in arcsec per pixel.')
    p.add_argument('--output_path', type=str,
                   default='/mnt/prep-data/prep_OBS',
                   help='path to save the converted maps.')
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # Load paths
    obs_paths = sorted(glob(args.obs_path))

    assert len(obs_paths) > 0, 'No files found.'

    # Load maps
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(_loadMLprepMap, zip(obs_paths, repeat(args.output_path), repeat(args.resolution)))
