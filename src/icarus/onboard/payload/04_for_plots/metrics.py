import math, os, torch
from pytorch_msssim import ms_ssim
import numpy as np
import torch

def check_range(a, str=""):
    if type(a) == np.ndarray:
        print(str+"type", a.dtype," range (min,mean,max)", np.min(a), np.mean(a), np.max(a))
    elif torch.is_tensor(a):
        print(str+"type", a.dtype," range (min,mean,max)", torch.min(a), torch.mean(a), torch.max(a))
    else:
        print(type(a))


def compute_psnr(a, b):
    if a.max() > 1. or b.max() > 1.:
        print("values > 1, maybe we should compute compute_psnr on data in the range between 0-1")
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
    #print("File", file_path, "has", convert_size(size_bytes))
    return size_bytes

def compression_rate(input_filename, compressed_filename):
    original_size = files_size(input_filename)
    latent_size = files_size(compressed_filename)

    reduction_factor = original_size / latent_size
    return reduction_factor
