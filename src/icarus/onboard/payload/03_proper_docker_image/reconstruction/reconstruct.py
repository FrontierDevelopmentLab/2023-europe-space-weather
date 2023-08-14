# File where we reconstruct the compressed image

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from torchvision import transforms
import torch

import math, os
from pytorch_msssim import ms_ssim


"""
/home/vitek/Vitek/Work/FDL23_HelioOnBoard/compress-ai-payload/03_proper_docker_image/results/latent_cor1_512_32_32.bytes
/home/vitek/Vitek/Work/FDL23_HelioOnBoard/compress-ai-payload/03_proper_docker_image/results/latent_cor2_2048_128_128.bytes
"""

# Cor 1 file
input_filename = "../data/cor1_20090822_130000_s4c1a.fts"
latent_name = "../results/latent_cor1_512_32_32"
# Cor 2 file
input_filename = "../data/cor2_20090611_125300_n4c2a.fts"
latent_name = "../results/latent_cor2_2048_128_128"


from compressai.zoo import bmshj2018_factorized
device = 'cpu'
net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)


orig_img = fits.getdata(input_filename)
print(orig_img.shape, orig_img.dtype)
print(np.min(orig_img), np.mean(orig_img), np.max(orig_img))
# plt.imshow(orig_img, cmap="gray")

img = orig_img.copy()

print("original data range (min,mean,max):", np.min(img), np.mean(img), np.max(img))  # 0-16k

img = np.asarray([img, img, img])  # fake rgb
img = np.transpose(img, (1, 2, 0)).astype(np.int16)

# normalise to [0, 255]
img = (img - img.min()) / (
        img.max() - img.min()
)
img = img.astype(np.float32)
print("normalised data range (min,mean,max):", np.min(img), np.mean(img), np.max(img))  # 0-1

x = transforms.ToTensor()(img)
x = x.unsqueeze(0).to('cpu')
print("x data range (min,mean,max):", torch.min(x), torch.mean(x), torch.max(x))  # 0-1

x_np = x.cpu().detach().numpy()
print(x_np.shape)

### Now load the saved files:

with open(latent_name+".bytes", "rb") as f:
    strings_loaded = f.read()
strings_loaded = [[strings_loaded]]

a, b = int(latent_name.split("_")[-2]), int(latent_name.split("_")[-1])
shape_loaded = ([a,b])

with torch.no_grad():
    out_net = net.decompress(strings_loaded, shape_loaded)
    #(is already called inside) out_net['x_hat'].clamp_(0, 1)

x_hat = out_net['x_hat']
print("x_hat data range (min,mean,max):", torch.min(x_hat), torch.mean(x_hat), torch.max(x_hat)) # 0-1

print(out_net.keys())

rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())



### And compare

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].title.set_text('Input')
ax[0].imshow(orig_img, cmap="gray")
ax[1].title.set_text('Decompressed from onxx model')
ax[1].imshow(rec_net, cmap="gray")
plt.show()



# 3 metrics

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

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

print("--- Metrics: ---")
print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}')
if 'likelihoods' in out_net.keys():
    print(f'Bit-rate: {compute_bpp(out_net):.3f} bpp')

original_size = files_size(input_filename)
latent_size = files_size(latent_name+".bytes")

reduction_factor = original_size / latent_size
print("Compressed with reduction factor by", round(reduction_factor,2), "times")

"""
Cor 1:
--- Metrics: ---
PSNR: 39.42dB
MS-SSIM: 0.9934
File ../data/cor1_20090822_130000_s4c1a.fts has 534.38 KB
File ../results/latent_cor1_512_32_32.bytes has 2.65 KB
Compressed with reduction factor by 201.77 times

Cor 2:
--- Metrics: ---
PSNR: 46.60dB
MS-SSIM: 0.9969
File ../data/cor2_20090611_125300_n4c2a.fts has 8.03 MB
File ../results/latent_cor2_2048_128_128.bytes has 41.01 KB
Compressed with reduction factor by 200.54 times
"""
