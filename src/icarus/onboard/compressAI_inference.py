import io
import math
import os

import astropy
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import torch
from compressai.zoo import bmshj2018_factorized
from ipywidgets import interact, widgets
from matplotlib import cm
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

plot_dir = "./src/icarus/plots/"
data_dir = "./src/icarus/data/"

if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"

    # load a pre-trained model
    net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)
    print(f"Parameters: {sum(p.numel() for p in net.parameters())}")

    # load a fits image
    fname = os.path.join(
        data_dir, "secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts"
    )
    img_data = fits.getdata(fname)
    # print(img_data.shape)  # 512 x 512
    arrays = [img_data, img_data.copy(), img_data.copy()]
    img_data_rgb = np.stack(arrays, axis=2).astype(np.int16)
    img_data_normalised = (img_data_rgb - img_data_rgb.min()) / (
        img_data_rgb.max() - img_data_rgb.min()
    )
    img_data_normalised = img_data_normalised.astype(np.float32)
    # print(img_data_rgb.dtype)
    # print(img_data_rgb[0,0,0])
    # print(img_data_rgb.min())
    # print(img_data_rgb.max())
    # print(img_data_rgb.shape)  # 512 x 512 x 3
    # img_data_rgb = Image.fromarray(img_data_rgb)
    # img_data_rgb = Image.fromarray(img_data)

    x = transforms.ToTensor()(img_data_normalised).unsqueeze(0).to(device)
    print(x.shape)
    print(x.nelement())  # 786432

    plotname = os.path.join(plot_dir, "secchi_1.png")
    plt.figure(figsize=(12, 9))
    plt.axis("off")
    plt.imshow(img_data_normalised)
    plt.show()
    plt.savefig(plotname)

    with torch.no_grad():
        out_net = net.forward(x)
    out_net["x_hat"].clamp_(0, 1)
    print(out_net.keys())

    rec_net = transforms.ToPILImage()(out_net["x_hat"].squeeze().cpu())

    plotname = os.path.join(plot_dir, "secchi_rec.png")
    plt.figure(figsize=(12, 9))
    plt.axis("off")
    plt.imshow(rec_net)
    plt.show()
    plt.savefig(plotname)


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

    # plotname = os.path.join(plot_dir, "secchi_2.png")
    # plt.show()
    # plt.savefig(plotname)

    # print(out_net["x_hat"].squeeze().cpu().nelement())  # 786432

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

    print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
    print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}')
    print(f"Bit-rate: {compute_bpp(out_net):.3f} bpp")
