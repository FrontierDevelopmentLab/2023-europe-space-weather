import numpy as np
import scipy
import matplotlib.pyplot as plt


stepnum = 43
stepnum_str = "0{}".format(stepnum) if stepnum < 100 else "{}".format(stepnum)
fname = "/mnt/ground-data/dens_stepnum_{}.sav".format(stepnum_str)

o = scipy.io.readsav(fname)

# dict_keys(['dens', 'r1d', 'th#1d', 'ph1d', 'stepnums', 'times'])
# dict_keys(['r1d', 'th1d', 'ph1d', 'dens', 'this_time'])
print(o.keys())
for i in o.values():
    if not isinstance(i, float):
        print(len(i))

dens = o['dens']  # (258, 128, 256)
shape = dens.shape

# axis coords
r  = o['r1d']  # 256
print(r)
th = o['th1d'] # 128 # minmax 0.32395396 2.8176386 # lat
ph = o['ph1d']  # 258 # minmax -3.1538644 3.1538644 # lon
print("th minmax", np.min(th), np.max(th))
print("ph minmax", np.min(ph), np.max(ph))

print("r minmax", np.min(r), np.max(r))
# 2D SLICE
# i_th = len(th) / 2
# theta = th[i_th]


plt.imshow(dens[:, 64, :], norm='log')
plt.savefig("theta_slice.jpg")
plt.close('all')



# polar plot: https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_demo.html
# https://stackoverflow.com/questions/17201172/a-logarithmic-colorbar-in-matplotlib-scatter-plot

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

rr, phph = np.meshgrid(r, ph, indexing = "ij")


x_vals = rr * np.cos(phph)
y_vals = rr * np.sin(phph)
z = np.transpose(dens[:, 64, :])

ax.set_rlim(21, 200)

ax.pcolormesh(phph, rr, z, edgecolors='face', norm='log')

ax.set_title("Density polar plot", va='bottom')
plt.show()
plt.savefig("polar_plot.jpg")
plt.close('all')


# 3D: SLOW
# ax = plt.figure().add_subplot(projection='3d')
# x_array = []
# y_array = []
# z_array = []
# c_array = []
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         for k in range(shape[2]):
#             radius = r[k]
#             theta = th[j]
#             phi = ph[i]
#             x = radius * np.sin(theta) * np.cos(phi)
#             y = radius * np.sin(theta) * np.sin(phi)
#             z = radius * np.cos(theta)
#             x_array.append(x)
#             y_array.append(y)
#             z_array.append(z)
#             c_array.append(dens[i, j, k])
# ax.scatter(x_array, y_array, z_array, c=c_array, cmap='coolwarm')
# # export to vtk for paraview?

# #plt.show()
# plt.savefig("density_cube.png")