import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from compressai.zoo import (  # “Variational Image Compression with a Scale Hyperprior”
    bmshj2018_factorized,
)
from pytorch_msssim import ms_ssim
from torchvision import transforms

# change name to 2023-europe-space-weather/src/icarus/data/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts
input_filename = "../data/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts"
orig_img = fits.getdata(input_filename)
print(orig_img.shape, orig_img.dtype)
print(np.min(orig_img), np.mean(orig_img), np.max(orig_img))
# plt.imshow(orig_img, cmap="gray")

# 1 compress
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"

# (default) quality=2 ~ 200x compression
net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)
print(f"Parameters: {sum(p.numel() for p in net.parameters())}")

img = orig_img.copy()

print(
    "original data range (min,mean,max):", np.min(img), np.mean(img), np.max(img)
)  # 0-16k

img = np.asarray([img, img, img])  # fake rgb
img = np.transpose(img, (1, 2, 0)).astype(np.int16)

# normalise to [0, 255]
img = (img - img.min()) / (img.max() - img.min())
img = img.astype(np.float32)
print(
    "normalised data range (min,mean,max):", np.min(img), np.mean(img), np.max(img)
)  # 0-1

x = transforms.ToTensor()(img)
x = x.unsqueeze(0).to(device)
print("x data range (min,mean,max):", torch.min(x), torch.mean(x), torch.max(x))  # 0-1

with torch.no_grad():
    # Full pass: out_net = net.forward(x)
    # Compress:
    print("x", x.shape)
    y = net.g_a(x)
    print("y", y.shape)
    y_strings = net.entropy_bottleneck.compress(y)
    print("len(y_strings) = ", len(y_strings[0]))

    strings = [y_strings]
    shape = y.size()[-2:]

print("for comparison, this is what we have now:")
print("compressed_strings", len(strings))
print("compressed_strings[0][0]", len(strings[0][0]))

print(type(strings[0][0]))
print(shape)
latent_name = "latent_" + str(shape[0]) + "_" + str(shape[1])

# Save compressed forms:
with open(latent_name + ".bytes", "wb") as f:
    f.write(strings[0][0])


# 2 decompress

with open(latent_name + ".bytes", "rb") as f:
    strings_loaded = f.read()
strings_loaded = [[strings_loaded]]

a, b = int(latent_name.split("_")[1]), int(latent_name.split("_")[2])
shape_loaded = [a, b]

with torch.no_grad():
    out_net = net.decompress(strings_loaded, shape_loaded)
    # (is already called inside) out_net['x_hat'].clamp_(0, 1)

x_hat = out_net["x_hat"]
print(
    "x_hat data range (min,mean,max):",
    torch.min(x_hat),
    torch.mean(x_hat),
    torch.max(x_hat),
)  # 0-1

print(out_net.keys())

rec_net = transforms.ToPILImage()(out_net["x_hat"].squeeze().cpu())
print(
    "reconstruction data range (min,mean,max):",
    np.min(rec_net),
    np.mean(rec_net),
    np.max(rec_net),
)  # 0-255 again

diff = torch.mean((out_net["x_hat"] - x).abs(), axis=1).squeeze().cpu()

fix, axes = plt.subplots(1, 3, figsize=(16, 12))
for ax in axes:
    ax.axis("off")

axes[0].imshow(img)
axes[0].title.set_text("Original")

axes[1].imshow(rec_net)
axes[1].title.set_text("Reconstructed")

im = axes[2].imshow(diff, cmap="viridis")
axes[2].title.set_text("Difference")
plt.colorbar(im, ax=axes[2])

plt.show()

# 3 metrics


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


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def files_size(file_path):
    size_bytes = os.path.getsize(file_path)
    print("File", file_path, "has", convert_size(size_bytes))
    return size_bytes


print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}')
if "likelihoods" in out_net.keys():
    print(f"Bit-rate: {compute_bpp(out_net):.3f} bpp")

original_size = files_size(input_filename)
latent_size = files_size(latent_name + ".bytes")

reduction_factor = original_size / latent_size
print("Compressed with reduction factor by", round(reduction_factor, 2), "times")
