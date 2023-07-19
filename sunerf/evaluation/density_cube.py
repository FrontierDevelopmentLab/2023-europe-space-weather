import numpy as np
import scipy

fname = "/mnt/ground-data/sample_dens_stepnum_43.sav"

o = scipy.io.readsav(fname)
print(o)