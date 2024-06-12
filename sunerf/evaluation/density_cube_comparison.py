import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import scipy
from dateutil.parser import parse

from sunerf.evaluation.loader import SuNeRFLoader

fname = "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube/dens_stepnum_045.sav"
chk_path = '/glade/work/rjarolim/sunerf-cme/all_v01/save_state.snf'
result_path = '/glade/work/rjarolim/sunerf-cme/all_v01/results'
os.makedirs(result_path, exist_ok=True)


# init loader
loader = SuNeRFLoader(chk_path, resolution=512)

o = scipy.io.readsav(fname)

date0 = parse("2010-04-03T09:04:00.000")
time = date0 + timedelta(hours=float(o['this_time']))
time = loader.normalize_datetime(time)

dens = o['dens']
ph = o['ph1d']
r = o['r1d']
th = o['th1d']

phi, theta, radius, t = np.meshgrid(ph + np.pi / 2, th, r, np.array([time]), indexing="ij")

x = radius * np.sin(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.sin(phi)
z = radius * np.cos(theta)

query_points = np.stack([x, y, z, t], axis=-1, dtype=np.float32)
query_points = query_points[:, :, :, 0]
model_out = loader.load_coords(query_points)
sunerf_density = model_out['density']

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})

rr, phph = np.meshgrid(r, ph, indexing="ij")

x_vals = rr * np.cos(phph)
y_vals = rr * np.sin(phph)


ax = axs[0]
ax.set_rlim(0, 100)
z = np.transpose(dens[:, 64, :])
pc = ax.pcolormesh(phph, rr, z, edgecolors='face', norm='log', cmap='inferno')
fig.colorbar(pc)
ax.set_title("Density polar", va='bottom')

ax = axs[1]
ax.set_rlim(0, 100)
z = np.transpose(sunerf_density[:, 64, :])
pc = ax.pcolormesh(phph, rr, z, edgecolors='face', norm='log', cmap='inferno', vmin=1e-10, vmax=1e-8)
fig.colorbar(pc)
ax.set_title("SuNeRF Density", va='bottom')

plt.show()
plt.savefig(os.path.join(result_path, f'comparison.jpg'), dpi=100)
plt.close('all')
