# (using pre-trained model for now)
# having encoder on neural compute stick
# decode on vm
# check performance is similar to encoder-decoder both on vm
# (using 2014-02-22 sequence)

import io
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from compressai.zoo import bmshj2018_factorized
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# read in original images again

eventdate = "20140222"
instrument = "cor2"  #'cor1'
satellite = "a"  # "b"
polar_angle = 0
framerate = 5

data_path = f"/mnt/onboard_data/data/{instrument}/{eventdate}_0*_n*{satellite}.fts"
compressed_dir = f"/mnt/onboard_data/compressed/output1"

input_image_path = (
    f"/mnt/onboard_data/visualization/cme_video_{instrument}_{satellite}/{polar_angle}/"
)

# general pre-trained model for inference (to be updated)
net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)

image_list = sorted(glob(os.path.join(input_image_path, "*.jpg")))

# TODO think about background reference image (always one static?)
# TODO process image first?

maes = []
mses = []

# iterate over all or choose one
for path in tqdm(image_list, total=len(image_list)):
    print(path)
    im = np.array(Image.open(path))
    x = transforms.ToTensor()(im).unsqueeze(0).to(device)
    print(x.shape)  # torch.Size([1, 3, 480, 640])

    # search for corresponding compressed
    compressed_fname = os.path.join(
        compressed_dir,
        "compressed_" + os.path.basename(path).replace(".jpg", ".npy"),
    )
    print(path, compressed_fname)
    y_compressed = np.load(compressed_fname)
    # print(y_compressed.shape)
    y_compressed = torch.tensor(y_compressed).to(device)  # tensor torch.float32
    print(type(y_compressed), y_compressed.shape, y_compressed.dtype)
    y_strings = net.entropy_bottleneck.compress(y_compressed)  # list
    print(type(y_strings), len(y_strings), type(y_strings[0]))
    strings = [y_strings]
    print(type(strings))
    shape = y_compressed.size()[-2:]  # torch.Size([1, 192, 30, 40])
    print("shape:", shape)

    # decompress
    with torch.no_grad():
        out_net = net.decompress(strings, shape)
    x_hat = out_net["x_hat"]

    # then compare
    mae = torch.mean((x_hat - x).abs()).squeeze().cpu()
    mse = torch.mean((x_hat - x) ** 2).squeeze().cpu()
    print("Metrics:", mae, mse)
    maes.append(mae)
    mses.append(mse)

    # break  # only run on one image

print(
    np.mean(maes), np.mean(mses)
)  # 0.008, 0.0009 before # 0.018415472 0.0012347241 now
