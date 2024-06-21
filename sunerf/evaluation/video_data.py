import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm

result_path = "/glade/work/rjarolim/sunerf-cme/ground_truth/video"
os.makedirs(result_path, exist_ok=True)

norm = LogNorm()

files_320 = sorted(glob.glob("/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view/dcmer_320W_bang_0000_pB_stepnum_*.fits"))
for i, f in enumerate(files_320):
    s_map = Map(f)
    f_path = os.path.join(result_path, f"320_{i:04d}.png")
    plt.imsave(f_path, norm(s_map.data), cmap=cm.soholasco2)


files_20 = sorted(glob.glob("/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view/dcmer_020W_bang_0000_pB_stepnum_*.fits"))
for i, f in enumerate(files_20):
    s_map = Map(f)
    f_path = os.path.join(result_path, f"020_{i:04d}.png")
    plt.imsave(f_path, norm(s_map.data), cmap=cm.soholasco2)


files_60 = sorted(glob.glob("/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_1view/dcmer_060W_bang_0000_pB_stepnum_*.fits"))
for i, f in enumerate(files_60):
    s_map = Map(f)
    f_path = os.path.join(result_path, f"060_{i:04d}.png")
    plt.imsave(f_path, norm(s_map.data), cmap=cm.soholasco2)
    print(f" 60W: Heliographic longitude {s_map.heliographic_longitude}, Carrington longitude {s_map.carrington_longitude}")