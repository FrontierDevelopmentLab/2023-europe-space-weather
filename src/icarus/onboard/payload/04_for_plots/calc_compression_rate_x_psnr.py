import numpy as np
import torch
import os
from glob import glob
from tqdm import tqdm
from baseline_compressor import baseline_compress
from data import data_there_v1
from ai_compressor import ai_compress
from metrics import compute_psnr, compute_msssim, check_range, compression_rate
from astropy.io import fits
import json


# input_folder = "/home/vitek/Vitek/Work/FDL23_HelioOnBoard/compress-ai-payload/data/cor1_data_random50"
input_folder = "/home/vitek/Vitek/Work/FDL23_HelioOnBoard/compress-ai-payload/data/cor2_data_random50"
# input_filename = "../data/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts"
#subset = 3 # up to 50 images...
subset = 50

cor12_str = input_folder.split("/")[-1][0:4]
results_name = "grid_results_"+cor12_str+"_from_"+str(subset)+".json"
print("Will save into >>", results_name)
values_for_quality = [1,2,3,4,5,6,7,8] # values between 1-8
values_for_compression = list(np.linspace(0, 400, 21)) # links to the desired psnr - 2x compression will be the psnr

verbose = False

def baseline(orig_x, input_filename, compression=1):
    # A baseline
    x = orig_x.astype(np.int16)
    input_image, x_min, x_max = data_there_v1(x)
    data = (255 * input_image).astype(np.uint8)
    reconstr, compressed_path = baseline_compress(data, c=compression, tmp_file_name="tmp_000.j2k")

    a = torch.from_numpy(input_image)
    if verbose: check_range(a, "Input image ~ ")

    b = reconstr.astype(np.float32) / 255.0
    b = torch.from_numpy(b)
    if verbose: check_range(b, "After j2k compression ~ ")

    psnr = compute_psnr(a, b)
    comp_rate = compression_rate(input_filename, compressed_path)
    return psnr, comp_rate

def model(orig_x, input_filename, quality=2):
    x = np.asarray([orig_x, orig_x, orig_x]).astype(np.int16)
    input_image, x_min, x_max = data_there_v1(x)
    data = torch.from_numpy(input_image).unsqueeze(0)

    reconstr, compressed_path = ai_compress(data, quality=quality, tmp_file_name="")

    a = torch.from_numpy(input_image[0,:,:]) # one band
    if verbose: check_range(a, "Input image ~ ")
    b = np.mean(reconstr, axis=0) # one band ~ avg from the 3 predicted bands
    b = torch.from_numpy(b/255.0)
    if verbose: check_range(b, "After AI compression ~ ")

    psnr = compute_psnr(a, b)
    comp_rate = compression_rate(input_filename, compressed_path)
    return psnr, comp_rate



all_image_paths = sorted(glob(os.path.join(input_folder, "*.fts")))
all_image_paths = all_image_paths[:subset]
print("will do", len(all_image_paths), "images")

grid_results = {}

for image_i, input_filename in enumerate(tqdm(all_image_paths)):
    print("image", image_i)
    grid_results[image_i] = {}

    orig_x = fits.getdata(input_filename).copy()

    # print("[Baseline]")
    for compression in values_for_compression:
        #compression = 95
        psnr, comp_rate = baseline(orig_x, input_filename, compression=compression)
        key = "base_"+str(compression).zfill(3)
        grid_results[image_i][key] = {}
        grid_results[image_i][key]["psnr"] = psnr
        grid_results[image_i][key]["comp_rate"] = comp_rate

    # print("PSNR", psnr, "\nCompression rate:", comp_rate)

    # B model
    # print("[AI model]")
    for quality in values_for_quality:
        #quality = 2
        psnr, comp_rate = model(orig_x, input_filename, quality=quality)
        key = "ai_"+str(quality).zfill(2)
        grid_results[image_i][key] = {}
        grid_results[image_i][key]["psnr"] = psnr
        grid_results[image_i][key]["comp_rate"] = comp_rate

        # print("PSNR", psnr, "\nCompression rate:", comp_rate)
        # print()


print(grid_results)
with open(results_name, 'w') as fp:
    json.dump(grid_results, fp)
