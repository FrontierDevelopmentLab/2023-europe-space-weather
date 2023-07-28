import glob
import logging
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
from dateutil.parser import parse
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from icarus.onboard.neural_compression.load_STEREO import load_data
from icarus.onboard.neural_compression.model import ResModel

results_path = "/mnt/neural_compression/version1"
os.makedirs(results_path, exist_ok=True)

dim = 128
n_layers = 4
max_epochs = 10

# 32 bits should be 4 bytes
# original_size = 2048 ** 2 * (72 / 3) * 4 --> 400 MB
# 868 KB
# compression --> 0.22 %

eventdate = "20140222"
instrument = "cor2"  #'cor1'
satellite = "a"  # "b"
polar_angle = 0
data_path = f"/mnt/onboard_data/data/{instrument}/{eventdate}_*_n*{satellite}.fts"

images, times = load_data(data_path, polar_angle)
images = np.clip(images, a_min=0, a_max=1)
images = np.moveaxis(
    images, 0, 2
)  # move "time" from first to last axis to be updated later

ref_time = min(times)
times = (times - ref_time) / timedelta(
    days=1
)  # times to floats, normalise to 1 day (depending on sequence)

num_channels = 1
cube_shape = (*images.shape, 1)  # x, y, t, c

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
n_gpus = torch.cuda.device_count()


model = ResModel(len(cube_shape) - 1, cube_shape[-1], dim, n_blocks=n_layers)
model_parallel = DataParallel(model)
model_parallel.to(device)

opt = torch.optim.Adam(model_parallel.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

mse_loss = nn.MSELoss(reduction="mean")

# TODO data loader from images and times from their filenames
# np_ds = NumpyDataset(data_mapping, time_mapping, spatial_norm, cube_shape, int(5e4))

coordinates = np.stack(
    np.mgrid[: images.shape[0], : images.shape[1], : images.shape[2]],
    -1,
    dtype=np.float32,
)  # (x, y, t, (3))

# replace with correct times
for i, t in enumerate(times):
    coordinates[:, :, i, 2] = t


batch_size = 2048 * 4

mask = ~np.isnan(images)
images_flat = images[mask]
coordinate_flat = coordinates[mask]

coordinate_tensor = torch.from_numpy(coordinate_flat).float()  # .reshape((-1, 3))
image_tensor = torch.from_numpy(images_flat).float().reshape((-1, 1))  # x * y * t, 1


# training
model_parallel.train()
for epoch in range(0, max_epochs):
    error_coords = []
    total_diff = []
    #
    for i in tqdm(range(image_tensor.shape[0] // batch_size + 1)):
        coordinate_batch = coordinate_tensor[i * batch_size : (i + 1) * batch_size]
        image_batch = image_tensor[i * batch_size : (i + 1) * batch_size]
        #
        opt.zero_grad()
        coordinate_batch, image_batch = coordinate_batch.to(device), image_batch.to(
            device
        )
        #
        prediction_batch = model_parallel(coordinate_batch)
        #
        loss = mse_loss(prediction_batch, image_batch)
        #
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parallel.parameters(), 0.1)
        opt.step()
        total_diff += [loss.detach().cpu().numpy()]
    #
    print(
        "[Epoch %05d/%05d] [Diff: %.08f]" % (epoch + 1, max_epochs, np.mean(total_diff))
    )

    torch.save({"model": model}, os.path.join(results_path, f"model.pt"))

    with torch.no_grad():
        prediction = []
        eval_coords_tensor = torch.from_numpy(coordinates).float().view(-1, 3)
        for i in tqdm(range(eval_coords_tensor.shape[0] // batch_size)):
            coordinate_batch = eval_coords_tensor[i * batch_size : (i + 1) * batch_size]
            prediction_batch = model_parallel(coordinate_batch.to(device))
            prediction += [prediction_batch.detach().cpu()]
        #
        image_seq = torch.cat(prediction).reshape(cube_shape[:-1]).numpy()
        for t in range(image_seq.shape[2]):
            image = image_seq[:, :, t]
            plt.subplot(121)
            plt.imshow(image, vmin=0, vmax=1)
            plt.subplot(122)
            plt.imshow(images[:, :, t], vmin=0, vmax=1)
            plt.savefig(os.path.join(results_path, f"{t:03d}.jpg"))
            plt.close()
