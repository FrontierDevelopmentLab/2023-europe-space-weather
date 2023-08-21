
import matplotlib.pyplot as plt
import math, os, torch
from torchvision import transforms
import numpy as np
from pytorch_msssim import ms_ssim
from compressai.zoo import bmshj2018_factorized # “Variational Image Compression with a Scale Hyperprior”
from astropy.io import fits


def ai_compress(data, quality=2, tmp_file_name=""):
    # change name to 2023-europe-space-weather/src/icarus/data/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts
    input_filename = "../data/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts"
    orig_img = fits.getdata(input_filename)
    # print(orig_img.shape, orig_img.dtype)
    # print(np.min(orig_img), np.mean(orig_img), np.max(orig_img))

    device = 'cpu'
    # (default) quality=2 ~ 200x compression
    net = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
    # print(f'Parameters: {sum(p.numel() for p in net.parameters())}')


    with torch.no_grad():
        # Full pass: out_net = net.forward(x)
        # Compress:
        # print("x", data.shape)
        y = net.g_a(data)
        # print("y", y.shape)
        y_strings = net.entropy_bottleneck.compress(y)
        # print("len(y_strings) = ", len(y_strings[0]))

        strings = [y_strings]
        shape = y.size()[-2:]

    # print("for comparison, this is what we have now:")
    # print("compressed_strings", len(strings))
    # print("compressed_strings[0][0]", len(strings[0][0]))

    # print(type(strings[0][0]))
    # print(shape)
    latent_name = "latent_" + str(shape[0]) + "_" + str(shape[1])

    # Save compressed forms:
    with open(latent_name + ".bytes", 'wb') as f:
        f.write(strings[0][0])

    # 2 decompress

    with open(latent_name + ".bytes", "rb") as f:
        strings_loaded = f.read()
    strings_loaded = [[strings_loaded]]

    a, b = int(latent_name.split("_")[1]), int(latent_name.split("_")[2])
    shape_loaded = ([a, b])

    with torch.no_grad():
        out_net = net.decompress(strings_loaded, shape_loaded)
        # (is already called inside) out_net['x_hat'].clamp_(0, 1)

    # x_hat = out_net['x_hat']
    # print("x_hat data range (min,mean,max):", torch.min(x_hat), torch.mean(x_hat), torch.max(x_hat))  # 0-1

    # print(out_net.keys())

    rec_net = 255.*out_net['x_hat'].squeeze().cpu().detach().numpy()

    # rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
    # print("reconstruction data range (min,mean,max):", np.min(rec_net), np.mean(rec_net),
    #       np.max(rec_net))  # 0-255 again

    output = rec_net
    path_to_compressed_file = latent_name + ".bytes"
    return output, path_to_compressed_file

