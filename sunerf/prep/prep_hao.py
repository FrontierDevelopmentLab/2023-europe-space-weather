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


def _load_HAO(file_path, occ_rad=0.1 * u.AU):
    """
    20 solar radii (0.1 AU) -->
    Dynamic model
    CMEs included
    """
    # assumes dsun is 1AU
    data, header = getdata(file_path), getheader(file_path)   

    # initialise 
    header['cunit1'] = 'arcsec'  
    header['cunit2'] = 'arcsec'  
    header['HGLN_OBS'] = np.rad2deg(header["OBS_LON"]) # rad to deg
    header['HGLT_OBS'] = np.rad2deg(header["OBS_LAT"])
    header['DSUN_OBS'] = (header["OBS_R0"]*u.Rsun).to("m").value # solar radii to m

    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"

    header["CDELT1"] = np.rad2deg(header["CDELT1"])*3600 # rad -- arcsec
    header["CDELT2"] = np.rad2deg(header["CDELT2"])*3600 # rad -- arcsec

    # manual set of centre value to 0
    header['CRVAL1'] = 0 
    header['CRVAL2'] = 0 

    header["CRPIX1"] = header["NAXIS1"]*0.5 + 0.5
    header["CRPIX2"] = header["NAXIS2"]*0.5 + 0.5

    #TODO: To be changed ... This is hardcoded for now, ask robert 
    header['wavelnth'] = 5200 

    map_i = Map(data, header)
    
    pixel_coords = all_coordinates_from_map(map_i)
    solar_center = SkyCoord(0*u.deg, 0*u.deg, frame=map_i.coordinate_frame)
    
    pixel_radii = np.sqrt((pixel_coords.Tx-solar_center.Tx)**2 + \
                      (pixel_coords.Ty-solar_center.Ty)**2)
    
    mask_inner = pixel_radii < map_i.rsun_obs*occ_rad.to(u.R_sun).value
 
    masked_i = Map(map_i.data, map_i.meta, mask=mask_inner)

    return masked_i

def _loadMLprepMap(file_path, out_path, resolution=None):
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
    s_map = _load_HAO(file_path)

    # adjust image size
    # if resolution:
    #     s_map = s_map.resample((resolution, resolution) * u.pix)

    # normalize image data
    data = s_map.data
    data[s_map.mask] = np.nan
    v_min, v_max = -18, -10
    data = (np.log(data) - v_min) / (v_max - v_min)
    data = data.astype(np.float32)

    s_map = Map(data, s_map.meta)
    dir_name = file_path.split(os.sep)[-2]
    s_map.save(os.path.join(out_path, dir_name + '_' + os.path.basename(file_path)), overwrite=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--hao_path', type=str,
                   default='/mnt/ground-data/data_fits/**/*.fits',
                   help='search path for AIA maps.')
    p.add_argument('--resolution', type=float,
                   default=1024,
                   help='target image scale in arcsec per pixel.')
    p.add_argument('--output_path', type=str,
                   default='/mnt/ground-data/prep_HAO',
                   help='path to save the converted maps.')
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # Load paths
    hao_paths = sorted(glob(args.hao_path))

    assert len(hao_paths) > 0, 'No files found.'

    # Load maps
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(_loadMLprepMap, zip(hao_paths, repeat(args.output_path), repeat(args.resolution)))