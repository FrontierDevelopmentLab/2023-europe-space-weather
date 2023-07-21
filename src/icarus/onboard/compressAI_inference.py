import io
import logging
import math
import os
import sys
from os.path import dirname

import astropy
import astropy.io.fits as fits

# Need to do sudo apt install python3-glymur to make this work! Then pip install glymur
import glymur
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from compressai.zoo import bmshj2018_factorized
from data_loader import FitsDataModule

# from ipywidgets import interact, widgets
from matplotlib import cm
from matplotlib.colors import LogNorm, PowerNorm
from PIL import Image
from pytorch_msssim import ms_ssim
from rich.progress import Progress
from torchvision import transforms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cwd = os.getcwd()
data_path = cwd
config_path = os.path.join(cwd, "..", "..", "..", "config")
with open(os.path.join(config_path, "onboard.yaml"), "r") as f:
    data_path = yaml.load(f, Loader=yaml.Loader)["drive_locations"]["datapath"]

ICARUS_DIR = dirname(dirname(__file__))
PLOT_DIR = os.path.join(cwd, "plots")
CME_DIR = os.path.join(PLOT_DIR, "cme")
OTHER_DIR = os.path.join(PLOT_DIR, "other")
DATA_DIR = os.path.join(
    data_path, "data", "cor2"
)  # moved "data" directory to /mnt/onboard_data/data/data
TEMP_DIR = os.path.join(PLOT_DIR, "temp")
one_image_test = False
"""
The following script runs a pre-trained compression model on a secchi fits file
by following: https://github.com/InterDigitalInc/CompressAI/blob/master/examples/CompressAI%20Inference%20Demo.ipynb
"""


def scale_minmax(image: np.ndarray) -> np.ndarray:
    """scale_minmax
        scales an image into a 0-1 range based on the images minimum and maximum, returns that image as float32
    Args:
        image (np.ndarray): image to rescale

    Returns:
        np.ndarray (float32): minmax rescaled image
    """
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    return image


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.0).item()


def compute_bpp(out_net):
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    ).item()


if __name__ == "__main__":
    if not os.path.exists(CME_DIR):
        os.makedirs(CME_DIR)
        print("The CME directory is created.")

    if not os.path.exists(OTHER_DIR):
        os.makedirs(OTHER_DIR)
        print("The other directory is created.")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print("Temp dir has been created.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # currently having the folloing error using cuda on onboard VM
    # RuntimeError: GET was unable to find an engine to execute this computation
    # device = "cpu"

    # load a pre-trained model
    net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)
    print(f"Parameters: {sum(p.numel() for p in net.parameters())}")

    if not one_image_test:
        # batch mode implementation ongoing
        fits_module = FitsDataModule()
        fits_module.setup("Testing")
        fits_loader = fits_module.val_dataloader()
        with Progress() as progress:
            task = progress.add_task("Batch Visualization", total=len(fits_loader))
            for batch in fits_loader:
                plt.close("all")
                images, event_index, fts_file = batch
                for i in range(images.size(0)):
                    image_np = np.log1p(
                        images[i].numpy().transpose(1, 2, 0)
                    )  # .astype(np.float32)

                    image_stretch = 255 * scale_minmax(image_np)
                    image_stretch = image_stretch.astype(np.uint8)

                    image_pil = Image.fromarray(image_stretch)
                    image_name = fts_file[i].split(".")[0]
                    initial_name = image_name + ".png"
                    is_cme = event_index[i]
                    dir_ = CME_DIR if is_cme else OTHER_DIR
                    plotname = os.path.join(dir_, initial_name)

                    image_pil.save(plotname)

                    x = images[i].to(device).unsqueeze(0) / 65535
                    # x = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)
                    # Compression and Decompression
                    with torch.no_grad():
                        out_net = net.forward(x)
                    out_net["x_hat"] += out_net["x_hat"].min()  # .clamp_(0, 1)
                    # print(out_net.keys())
                    # print(out_net["x_hat"].shape)

                    rec_net = (
                        out_net["x_hat"]
                        .squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                        .transpose((1, 2, 0))
                    )
                    reconstructed_name = image_name + "_reconstructed.png"
                    plotname = os.path.join(dir_, reconstructed_name)
                    plt.figure(figsize=(12, 9))
                    plt.axis("off")
                    plt.imshow(scale_minmax(rec_net))
                    plt.show()
                    plt.savefig(plotname)

                    # comparison
                    diff = (
                        torch.mean((out_net["x_hat"] - x).abs(), axis=1)
                        .squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    fix, axes = plt.subplots(1, 5, figsize=(16, 12))
                    for ax in axes:
                        ax.axis("off")

                    im = axes[0].imshow(
                        image_np[:, :, 0]
                    )  # Check the shape, should be the raw image
                    # axes[0].imshow(image_pil)
                    axes[0].title.set_text(
                        "Original - CME present: {}".format(
                            "Yes" if is_cme == 1 else "No"
                        )
                    )

                    # Estimate bits per pixel
                    n_pixels = np.product(image_np[:, :, 0].shape)
                    # To test bitrate, save image in J2K, check filesize of that, then delete the file again
                    full_file_name = (
                        TEMP_DIR + "/" + image_name + ".j2k"
                    )  # DATA_DIR + "/" + fts_file[i] #J2K file
                    jp2 = glymur.Jp2k(
                        full_file_name, data=image_stretch[:, :, 0].astype(np.uint8)
                    )  # set grayscale J2K
                    jp2_x = (
                        torch.from_numpy(jp2[:]).to(device).unsqueeze(0) / 65535
                    )  # Similar setup to x
                    j2k_recon_diff = (
                        torch.mean((jp2_x - x).abs(), axis=1)
                        .squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                    )

                    # Assume that the data is saved next to the header, so filesize is approximately the size of the file in bytes + size of image in bytes. Therefore, n_bits = 8*(filesize - headersize)
                    n_bits_estimate = 8 * (os.stat(full_file_name).st_size)
                    original_bits_per_pixel = np.nan
                    if n_pixels > 0:
                        original_bits_per_pixel = n_bits_estimate / n_pixels

                    # fix.colorbar(im, ax=axes[0])
                    # m.plot(norm=PowerNorm(0.5, vmin=0, vmax=1e-7, clip=True))
                    im = axes[1].imshow(
                        rec_net[:, :, 0]
                    )  # Should be OK, just dark as the range is large
                    axes[1].title.set_text(
                        "Reconstructed with VQVAE \n PSNR: {:.2f} dB \n MS-SSIM: {:.4f} \n Bits per Pixel {:.3f}".format(
                            compute_psnr(x, out_net["x_hat"]),
                            compute_msssim(x, out_net["x_hat"]),
                            compute_bpp(out_net),
                        )
                    )
                    # fix.colorbar(im, ax=axes[1])
                    # Compute MSE
                    mse = (
                        torch.mean((out_net["x_hat"] - x) ** 2, axis=1)
                        .squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                        .mean()
                    )
                    im = axes[2].imshow(diff, cmap="viridis")
                    axes[2].title.set_text("Difference - MSE: {:.6f}".format(mse))
                    # fix.colorbar(im, ax=axes[2])
                    # plot comparison of original, reconstructed and diff

                    im = axes[3].imshow(jp2[:])  # Reconstructed in JP2 format
                    x_red = x[:, 0, :, :].unsqueeze(0)
                    jp2_x_inc = jp2_x.unsqueeze(0)
                    axes[3].title.set_text(
                        "Reconstructed From J2K \n PSNR: {:.2f} dB \n MS-SSIM: {:.4f} \n Bits per Pixel {:.3f}".format(
                            compute_psnr(x, jp2_x),
                            compute_msssim(x_red, jp2_x_inc),
                            original_bits_per_pixel,
                        )
                    )  # Need to modify x in mssim due to expected data shape
                    # fix.colorbar(im, ax=axes[1])
                    # Compute MSE For J2K
                    mse_j2k = (
                        torch.mean((jp2_x - x[:, 0, :, :]) ** 2, axis=1)
                        .squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                        .mean()
                    )
                    im = axes[4].imshow(j2k_recon_diff, cmap="viridis")
                    axes[4].title.set_text(
                        "Difference J2K - MSE: {:.6f}".format(mse_j2k)
                    )
                    # fix.colorbar(im, ax=axes[2])
                    # plot comparison of original, reconstructed and diff
                    compare_name = image_name + "_comparison.png"
                    plotname = os.path.join(dir_, compare_name)
                    plt.show()
                    plt.savefig(plotname)

                progress.update(task, advance=1)  # update progress bar

    else:
        # load a fits image
        fname = os.path.join(
            ICARUS_DIR,
            "data",
            "secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts",
        )
        img_data = fits.getdata(fname)
        # print(img_data.shape)  # 512 x 512

        # our input is gray-scaled; stack the same input three times to fake a rgb image
        arrays = [img_data, img_data.copy(), img_data.copy()]
        img_data_rgb = np.stack(arrays, axis=2).astype(np.int16)
        # normalise to [0, 1]
        img_data_normalised = (img_data_rgb - img_data_rgb.min()) / (
            img_data_rgb.max() - img_data_rgb.min()
        )
        img_data_normalised = img_data_normalised.astype(np.float32)
        # print(img_data_rgb.shape)  # 512 x 512 x 3

        # plot the original fits image as png
        plotname = os.path.join(PLOT_DIR, "secchi_original.png")
        plt.figure(figsize=(12, 9))
        plt.axis("off")
        plt.imshow(img_data_normalised)
        plt.show()
        plt.savefig(plotname)

        x = transforms.ToTensor()(img_data_normalised).unsqueeze(0).to(device)
        # print(x.nelement())  # 786432

        # # compress (also done in: out_net = net.forward(x))
        # with torch.no_grad():
        #     print("x", x.shape)
        #     y = net.g_a(x)
        #     print("y", y.shape)
        #     y_strings = net.entropy_bottleneck.compress(y)
        #     print("len(y_strings)=", len(y_strings))

        # strings = [y_strings]
        # shape = y.size()[-2:]

        # # decompress (also done in: out_net = net.forward(x))
        # with torch.no_grad():
        #     out_net = net.decompress(strings, shape)
        # x_hat = out_net["x_hat"]

        # compress and decompress
        with torch.no_grad():
            out_net = net.forward(x)
        # out_net["x_hat"].clamp_(0, 1)
        # print(out_net.keys())

        # print(out_net["x_hat"].squeeze().cpu().nelement())  # 786432

        # save reconstructed image
        rec_net = transforms.ToPILImage()(out_net["x_hat"].squeeze().cpu())

        plotname = os.path.join(PLOT_DIR, "secchi_reconstructed.png")
        plt.figure(figsize=(12, 9))
        plt.axis("off")
        plt.imshow(rec_net)
        plt.show()
        plt.savefig(plotname)

        # comparison
        diff = torch.mean((out_net["x_hat"] - x).abs(), axis=1).squeeze().cpu()

        fix, axes = plt.subplots(1, 3, figsize=(16, 12))
        for ax in axes:
            ax.axis("off")

        axes[0].imshow(img_data_normalised)
        axes[0].title.set_text("Original")

        axes[1].imshow(rec_net)
        axes[1].title.set_text("Reconstructed")

        axes[2].imshow(diff, cmap="viridis")
        axes[2].title.set_text("Difference")

        # plot comparison of original, reconstructed and diff
        plotname = os.path.join(PLOT_DIR, "secchi_comparison.png")
        plt.show()
        plt.savefig(plotname)

        with torch.no_grad():
            # Compress:
            print("x", x.shape)
            y = net.g_a(x)
            print("y", y.shape)
            y_strings = net.entropy_bottleneck.compress(y)
            print("len(y_strings) = ", len(y_strings[0]))

            strings = [y_strings]
            shape = y.size()[-2:]

        with open("latents.bytes", "wb") as f:
            f.write(strings[0][0])

    # compute metrics

    print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
    print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}')
    print(f"Bit-rate: {compute_bpp(out_net):.3f} bpp")
