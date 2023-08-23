import os

import numpy as np
import torch
from tqdm import tqdm
import scipy
from sunpy.map import Map
from datetime import datetime
import pickle
import json

from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.utilities.data_loader import normalize_datetime

'''
python -m sunerf.evaluation.density_cube_eval
'''

START_STEPNUM = 5
END_STEPNUM = 74
CHUNKS = 4

# R_SUN_CM = 6.957e+10
# GRID_SIZE = 500 / 16  # solar radii

ignore_half_of_r = True

# og-eda and og-eda-3
ckpt_dirname = "HAO_pinn_cr_2view_a26978f_heliographic_reformat"
# og-eda
# ckpt_dirname = "HAO_pinn_2view_no_physics"
# ckpt_dirname = "HAO_pinn_2view_cr"
# ckpt_dirname = "HAO_pinn_5view_cr"
# ckpt_dirname = "HAO_pinn_5view_no_physics"
# og-eda-3
# ckpt_dirname = "HAO_pinn_2view_cr3"
# ckpt_dirname = "HAO_pinn_1view_cr3"
# ckpt_dirname = "HAO_pinn_5view_cr3"
# scan
# ckpt_dirname = "HAO_pinn_1view_no_physics"
# ckpt_dirname = "HAO_pinn_1view_cr"
# ckpt_dirname = "HAO_pinn_5view_c"


def save_stepnum_to_datetime():
    stepnum_to_datetime = dict()

    for stepnum in range(5, 80, 1):
        map_path = "/mnt/prep-data/prep_HAO_2view/dcmer_280W_bang_0000_pB_stepnum_%03d.fits" % stepnum
        s_map = Map(map_path)
        dt = s_map.date.datetime
        stepnum_to_datetime[stepnum] = dt.strftime("%Y-%m-%d %H:%M:%S")

    print(stepnum_to_datetime)
    with open('/mnt/ground-data/stepnum_to_datetime.pkl', 'wb') as f:
        pickle.dump(stepnum_to_datetime, f)

def load_stepnum_to_datetime():
    with open('/mnt/ground-data/stepnum_to_datetime.pkl', 'rb') as f:
        stepnum_to_datetime = pickle.load(f)
    return stepnum_to_datetime

# load datetime for each stepnum
stepnum_to_datetime = load_stepnum_to_datetime()
# convert datetime from string to datetime.datetime
def dtstr_to_datetime(dtstr):
    return datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S")
stepnum_to_datetime = dict(map(lambda kv: (kv[0], dtstr_to_datetime(kv[1])), stepnum_to_datetime.items()))

mae_all_stepnums = []

for stepnum in tqdm(range(START_STEPNUM, END_STEPNUM + 1, 1)):

    # load ground truth
    gt_fname = "/mnt/ground-data/density_cube/dens_stepnum_%03d.sav" % stepnum
    o = scipy.io.readsav(gt_fname)
    ph = o['ph1d']  # (258,)
    th = o['th1d']  # (128,)
    
    # ignore half of r
    if ignore_half_of_r:
        r_size = len(o['r1d'])
        r = o['r1d'][:int(r_size / 2)]  # (256,) -> (128, 0)
        density_gt = o['dens'][:,:,:int(r_size / 2)]  # (258, 128, 256) (phi, theta, r)
    else:
        r = o['r1d']  # (256,)
        density_gt = o['dens']  # (258, 128, 256) (phi, theta, r)

    # load model checkpoint
    base_path = '/mnt/training/' + ckpt_dirname
    chk_path = os.path.join(base_path, 'save_state.snf')
    loader = SuNeRFLoader(chk_path, resolution=512)

    # put th into chunks to avoid CUDA out of memory error
    th = th.reshape(CHUNKS, -1)

    time = normalize_datetime(stepnum_to_datetime[stepnum])
    observer_offset = np.deg2rad(90)
    ph_copy = ph.copy() + observer_offset

    density_chunks = []
    for chunk in tqdm(range(CHUNKS)):
        phph, thth, rr = np.meshgrid(ph_copy, th[chunk], r, indexing = "ij")

        x = rr * np.cos(phph) * np.sin(thth)
        y = rr * np.sin(phph) * np.sin(thth)
        z = rr * np.cos(thth)
        t = np.ones_like(rr) * time
        query_points_npy = np.stack([x, y, z, t], -1).astype(np.float32)  # (258, 32, 256, 4) for one chunk

        query_points = torch.from_numpy(query_points_npy)
        enc_query_points = loader.encoding_fn(query_points.view(-1, 4))

        # model inference
        with torch.no_grad():  # required for memory to be cleared properly
            raw = loader.fine_model(enc_query_points)
        density = raw[..., 0]
        # density = 10 ** (15 + raw[..., 0])
        density = density.view(query_points_npy.shape[:3]).cpu().detach().numpy()
        density_chunks.append(density)

    density = np.concatenate(density_chunks, 1)  # in electrons / r_sun
    # convert unit to that of ground truth: electrons / cm^3
    # density *= GRID_SIZE # electrons / grid cell
    # density *= (GRID_SIZE * R_SUN_CM) ** (-3)  # electrons / cm^3
    # density *= GRID_SIZE ** (-2) * R_SUN_CM ** (-3)

    # compare density to ground truth
    rel_density = density / np.median(density)
    rel_density_gt = density_gt / np.median(density_gt)

    print(rel_density[0])
    print(rel_density_gt[0])
    mae = (np.abs(rel_density - rel_density_gt)).mean(axis=None)
    print(mae)

    mae_all_stepnums.append(mae)

print(mae_all_stepnums)
mae_avg = sum(mae_all_stepnums) / len(mae_all_stepnums)
print(mae_avg)


# save eval to json
output_fname = "eval_half.json" if ignore_half_of_r else "eval.json"
eval_dict = {"mae_all_stepnums": mae_all_stepnums, "mae_avg": mae_avg}
with open(os.path.join(base_path, output_fname), 'w') as fp:
    json.dump(eval_dict, fp)
