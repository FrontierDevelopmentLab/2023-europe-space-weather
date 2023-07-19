import numpy as np
import scipy
import matplotlib.pyplot as plt

fname = "/mnt/ground-data/sample_dens_stepnum_43.sav"

o = scipy.io.readsav(fname)

# dict_keys(['dens', 'r1d', 'th1d', 'ph1d', 'stepnums', 'times'])
print(o.keys())
for i in o.values():
    print(len(i))

dens = o['dens']  # (258, 128, 256)
shape = dens.shape

# axis coords
r  = o['r1d']  # 256
th = o['th1d'] # 128 # minmax 0.32395396 2.8176386 # lat
ph = o['ph1d']  # 258 # minmax -3.1538644 3.1538644 # lon
print("th minmax", np.min(th), np.max(th))
print("ph minmax", np.min(ph), np.max(ph))

# 2D SLICE
# i_th = len(th) / 2
# theta = th[i_th]


plt.imshow(dens[:, 64, :], norm='log')
plt.savefig("theta_slice.jpg")
plt.close('all')


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)


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