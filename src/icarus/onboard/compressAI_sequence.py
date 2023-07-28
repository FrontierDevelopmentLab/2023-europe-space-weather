import io
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from compressai.zoo import bmshj2018_factorized
from neural_compression.load_STEREO import load_data
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

eventdate = "20140222"
instrument = "cor2"  #'cor1'
satellite = "a"  # "b"
polar_angle = 0
framerate = 5
data_path = f"/mnt/onboard_data/data/{instrument}/{eventdate}_0*_n*{satellite}.fts"
video_name = f"/mnt/onboard_data/visualization/compressed_bmshj2018_factorized/{eventdate}_{polar_angle}.mp4"
video_diff_name = f"/mnt/onboard_data/visualization/compressed_bmshj2018_factorized/{eventdate}_{polar_angle}_diff.mp4"

input_image_path = (
    f"/mnt/onboard_data/visualization/cme_video_{instrument}_{satellite}/{polar_angle}/"
)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
denormalise = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)
net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)

image_list = sorted(glob(os.path.join(input_image_path, "*.jpg")))
im_size = np.array(Image.open(image_list[0])).shape
video = cv2.VideoWriter(video_name, fourcc, framerate, (im_size[1], im_size[0]))
video_diff = cv2.VideoWriter(
    video_diff_name, fourcc, framerate, (im_size[1], im_size[0])
)

for path in image_list:
    im = np.array(Image.open(path))
    x = transforms.ToTensor()(im).unsqueeze(0).to(device)
    # x = normalise(x)
    # print(x.max())
    # print(x.min())

    with torch.no_grad():
        out_net = net.forward(x)
        out_im = out_net["x_hat"].clip(0, 1)
        compressed_im = transforms.ToPILImage()(out_im.squeeze().cpu())

        # print(out_im.max())
        # print(out_im.min())

        diff = (out_net["x_hat"] - x).abs().clip(0, 1).squeeze().cpu()
        diff = transforms.ToPILImage()(1 - diff)
        video.write(np.array(compressed_im).astype(np.uint8)[..., [2, 1, 0]])
        video_diff.write(np.array(diff).astype(np.uint8)[..., [2, 1, 0]])

cv2.destroyAllWindows()
video.release()
video_diff.release()


if (
    False
):  # This is compression from raw data, couldn't find a correct vmin, vmax range for saving outputs.
    images, times = load_data(data_path, polar_angle)

    video = cv2.VideoWriter(
        video_name, fourcc, framerate, (480, 640)
    )  # TODO: change hard-coded dimensions to dynamic

    fig = plt.figure()

    for im in images:
        min_ = np.nanmin(im)
        max_ = np.nanmax(im)
        image = np.zeros((im.shape[0], im.shape[1], 3))
        image = image + im.reshape(im.shape[0], im.shape[1], 1)
        im_normalised = (image - min_) / (max_ - min_)
        im_normalised = im_normalised.astype(np.float32)
        x = transforms.ToTensor()(im_normalised).unsqueeze(0).to(device)

        with torch.no_grad():
            out_net = net.forward(x)
            compressed_im = transforms.ToPILImage()(out_net["x_hat"].squeeze().cpu())
            diff = torch.mean((out_net["x_hat"] - x).abs(), axis=1).squeeze().cpu()
            compressed_im = compressed_im * (max_ - min_) + min_
            print(np.nanmax(compressed_im))
            print(np.nanmin(compressed_im))
            plt.imshow(compressed_im, origin="lower", cmap="stereocor2", vmin=0)
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format="raw", dpi=100)
            io_buf.seek(0)
            img_out = np.reshape(
                np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
            )

            print(img_out.shape)
            io_buf.close()
            video.write(img_out[:, :, :-1])
            plt.close()
    cv2.destroyAllWindows()
    video.release()
