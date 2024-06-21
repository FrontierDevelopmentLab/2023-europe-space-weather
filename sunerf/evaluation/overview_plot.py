import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm

s_map = Map(
    "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2view/dcmer_020W_bang_0000_pB_stepnum_057.fits")
cmap = cm.soholasco2.copy()
plt.imsave("/glade/work/rjarolim/sunerf-cme/ground_truth/coronagraph.png", LogNorm()(s_map.data), cmap=cmap)

euv_map = Map("/glade/work/rjarolim/sunerf-cme/ground_truth/aia.lev1_euv_12s.2014-04-13T000008Z.193.image_lev1.fits")
euv_map.plot()
plt.savefig("/glade/work/rjarolim/sunerf-cme/ground_truth/test.png")

coords = all_coordinates_from_map(euv_map)
radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / euv_map.rsun_obs

alpha_mask = (2 - radius ** 4)
alpha_mask = np.clip(alpha_mask, a_min=0, a_max=1)

norm = euv_map.plot_settings['norm']
cmap = euv_map.plot_settings['cmap']
img = plt.get_cmap(cmap)(norm(euv_map.data))
img[:, :, -1] = alpha_mask

plt.imsave("/glade/work/rjarolim/sunerf-cme/ground_truth/euv.png", img)
