# (using pre-trained model for now)
# encode on vm
# decode on same vm
# check performance
# use 2014-02-22 sequence

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

eventdate = "20140222"
instrument = "cor2"  #'cor1'
satellite = "a"  # "b"
polar_angle = 0
framerate = 5

data_path = f"/mnt/onboard_data/data/{instrument}/{eventdate}_0*_n*{satellite}.fts"

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

    # x is between 0 and 1
    # print(x.max())
    # print(x.min())

    # this does both encoding and decoding
    # with torch.no_grad():
    # out_net = net.forward(x) # encoding and decoding?
    # out_im = out_net["x_hat"].clip(0, 1)
    # compressed_im = transforms.ToPILImage()(out_im.squeeze().cpu())

    # encode/compress
    with torch.no_grad():
        y = net.g_a(x)
        print("y", y.shape, y.dtype)
        y_strings = net.entropy_bottleneck.compress(y)
        print("y_strings", type(y_strings))
        print("len(y_strings)=", len(y_strings))
        strings = [y_strings]
        shape = y.size()[-2:]
        # print(type(y_strings), y_strings.shape, y_strings.dtype)

    # decompress
    with torch.no_grad():
        out_net = net.decompress(strings, shape)
    x_hat = out_net["x_hat"]

    mae = torch.mean((x_hat - x).abs()).squeeze().cpu()
    mse = torch.mean((x_hat - x) ** 2).squeeze().cpu()
    print("Metrics:", mae, mse)
    maes.append(mae)
    mses.append(mse)

    # break  # only run on one image

print(np.mean(maes), np.mean(mses))  # 0.008, 0.0009
