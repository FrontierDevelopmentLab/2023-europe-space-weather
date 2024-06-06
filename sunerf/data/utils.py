from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch
from sunpy.visualization.colormaps import cm

sdo_img_norm = ImageNormalize(vmin=0, vmax=1, stretch=LinearStretch(), clip=True)

# !stretch is connected to NeRF!
sdo_norms = {171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=False),
             193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=False),
             195: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=False),
             211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=False),
             284: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=False),
             304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.005), clip=False), }

psi_norms = {171: ImageNormalize(vmin=0, vmax=22348.267578125, stretch=AsinhStretch(0.005), clip=True),
             193: ImageNormalize(vmin=0, vmax=50000, stretch=AsinhStretch(0.005), clip=True),
             211: ImageNormalize(vmin=0, vmax=13503.1240234375, stretch=AsinhStretch(0.005), clip=True), }

so_norms = {304: ImageNormalize(vmin=0, vmax=300, stretch=AsinhStretch(0.005), clip=False),
            174: ImageNormalize(vmin=0, vmax=300, stretch=AsinhStretch(0.005), clip=False)}

sdo_cmaps = {171: cm.sdoaia171, 174: cm.sdoaia171, 193: cm.sdoaia193, 211: cm.sdoaia211, 304: cm.sdoaia304}
