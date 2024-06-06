import numpy as np
import scipy
import matplotlib.pyplot as plt
from glob import glob
import os
import re

from mpl_toolkits.axes_grid1 import make_axes_locatable

# stepnum = 43
# stepnum_str = "0{}".format(stepnum) if stepnum < 100 else "{}".format(stepnum)
# fname = "/mnt/ground-data/dens_stepnum_{}.sav".format(stepnum_str)

# o = scipy.io.readsav(fname)

# # dict_keys(['dens', 'r1d', 'th#1d', 'ph1d', 'stepnums', 'times'])
# # dict_keys(['r1d', 'th1d', 'ph1d', 'dens', 'this_time'])
# print(o.keys())
# for i in o.values():
#     if not isinstance(i, float):
#         print(len(i))

# dens = o['dens']  # (258, 128, 256)
# shape = dens.shape

# # axis coords
# r  = o['r1d']  # 256
# print(r)
# th = o['th1d'] # 128 # minmax 0.32395396 2.8176386 # lat
# ph = o['ph1d']  # 258 # minmax -3.1538644 3.1538644 # lon
# print("th minmax", np.min(th), np.max(th))
# print("ph minmax", np.min(ph), np.max(ph))

# print("r minmax", np.min(r), np.max(r))
# # 2D SLICE
# # i_th = len(th) / 2
# # theta = th[i_th]

# plt.imshow(dens[:, 64, :], norm='log')
# plt.savefig("theta_slice.jpg")
# plt.close('all')


# polar plot: https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_demo.html
# https://stackoverflow.com/questions/17201172/a-logarithmic-colorbar-in-matplotlib-scatter-plot

savefile_folder = "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube"
result_path = '/glade/work/rjarolim/sunerf-cme/ground_truth'
os.makedirs(result_path, exist_ok=True)

for fname in glob(savefile_folder + "/*_0*.sav"):
    stepnum = int(re.findall(r'\d+',os.path.basename(fname))[-1])
    stepnum = "0{}".format(stepnum) if stepnum < 100 else "{}".format(stepnum)

    try:
        o = scipy.io.readsav(fname)
    except:
        print(f"Error reading {fname}")
        continue

    dens = o['dens']  # (258, 128, 256)
    ph = o['ph1d']
    r  = o['r1d']

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    rr, phph = np.meshgrid(r, ph, indexing = "ij")

    x_vals = rr * np.cos(phph)
    y_vals = rr * np.sin(phph)
    z = np.transpose(dens[:, 64, :])

    ax.set_rlim(21, 200)

    pc = ax.pcolormesh(phph, rr, z, edgecolors='face', norm='log', cmap='inferno')
    fig.colorbar(pc)

    ax.set_title("Density polar plot", va='bottom')
    plt.show()
    plt.savefig(os.path.join(result_path, f'dens_polar_{stepnum}.jpg'), dpi=100)
    plt.close('all')
