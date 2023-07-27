import argparse
import multiprocessing
import os
from glob import glob
from itertools import repeat

import numpy as np
from astropy import units as u
from astropy.io.fits import getheader, getdata
from sunpy.map import Map
from sunpy.map.maputils import all_coordinates_from_map
from astropy.coordinates import SkyCoord


def _loadMLprepMap(file_path, out_path, resolution, occ_rad=1.015*u.R_sun):
    """Load and preprocess PSI file.


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels.

    Returns
    -------
    the preprocessed SunPy Map
    """
    # load Map
    s_map = Map(file_path)

    # change resolution
    s_map = s_map.resample((resolution, resolution) * u.pix)

    pixel_coords = all_coordinates_from_map(s_map)
    solar_center = SkyCoord(0*u.deg, 0*u.deg, frame=s_map.coordinate_frame)
    
    pixel_radii = np.sqrt((pixel_coords.Tx-solar_center.Tx)**2 + \
                      (pixel_coords.Ty-solar_center.Ty)**2)
    
    mask = pixel_radii < s_map.rsun_obs*occ_rad.to(u.R_sun).value
    
    # normalize image data
    data = s_map.data
    data[mask] = np.nan
    v_min, v_max = -31, -13 # full value range: -31, -13; -18, -10
    data = (np.log(data) - v_min) / (v_max - v_min)
    data = data.astype(np.float32)

    s_map = Map(data, s_map.meta)
    s_map.save(os.path.join(out_path, os.path.basename(file_path)), overwrite=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--psi_path', type=str,
                   default='/mnt/ground-data/PSI/b_raw/*.fits',
                   help='search path for AIA maps.')
    p.add_argument('--resolution', type=float,
                   default=512,
                   help='target image scale in arcsec per pixel.')
    p.add_argument('--output_path', type=str,
                   default='/mnt/ground-data/prep_PSI/b_raw',
                   help='path to save the converted maps.')
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # Load paths
    psi_paths = sorted(glob(args.psi_path))

    assert len(psi_paths) > 0, 'No files found.'

    # Load maps
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(_loadMLprepMap, zip(psi_paths, repeat(args.output_path), repeat(args.resolution)))
