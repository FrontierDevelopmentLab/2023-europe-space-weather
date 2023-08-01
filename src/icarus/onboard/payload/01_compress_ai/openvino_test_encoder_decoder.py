from pathlib import Path

# Works with older 2021 version:
import openvino
import openvino.inference_engine
from openvino.inference_engine import IECore

import os
from glob import glob
import logging
import sys
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from torchvision import transforms
from PIL import Image

# 1 load openvino model


class OpenVinoModel:
    def __init__(self, ie, model_path, batch_size=1):
        self.logger = logging.getLogger("model")
        print("\tModel path: {}".format(model_path))
        self.net = ie.read_network(model_path, model_path[:-4] + ".bin")
        self.set_batch_size(batch_size)

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def set_batch_size(self, batch):
        shapes = {}
        for input_layer in self.net.input_info:
            new_shape = [batch] + self.net.input_info[input_layer].input_data.shape[1:]
            shapes.update({input_layer: new_shape})
        self.net.reshape(shapes)

def device_available():
    ie = IECore()
    devs = ie.available_devices
    return "MYRIAD" in devs


def load_model(model_path, device="MYRIAD") -> Callable:
    # Broadly copied from the OpenVINO Python examples
    ie = IECore()
    try:
        ie.unregister_plugin("MYRIAD")
    except:
        pass

    model = OpenVinoModel(ie, model_path)
    tic = time.time()
    print("Loading ONNX network to ", device, "...")

    exec_net = ie.load_network(
        network=model.net, device_name=device, config=None, num_requests=1
    )
    toc = time.time()
    print("one, time elapsed : {} seconds".format(toc - tic))

    def predict(x: np.ndarray) -> np.ndarray:
        """
        Predict function using the myriad chip

        Args:
            x: (C, H, W) 3d tensor

        Returns:
            (C, H, W) 3D network logits

        """
        print(x.shape)

        result = exec_net.infer({"input": x[np.newaxis]})
        return result["output"][0]  # (n_class, H, W)

    return predict


BATCH_SIZE = 1

example_input = np.random.rand(BATCH_SIZE, 3, 480, 640)
# model_path = "onboard_net.onnx"
model_path = "onboard_compressor_y.onnx"
image_dir = "/home/chiaman/workspace/2023-europe-space-weather/data/cme_20140222_cor2_a_0/"
output_dir = "/home/chiaman/workspace/2023-europe-space-weather/data/output/"

# device = "CPU"
device = "MYRIAD"
ie = IECore()
model = OpenVinoModel(ie, model_path)
exec_net = ie.load_network(
    network=model.net, device_name=device, config=None, num_requests=1
)

result = exec_net.infer({"input": example_input})

for output in list(exec_net.outputs.keys()):
    print("output named", output, "gives", result[output].shape)

image_list = sorted(glob("{}*.jpg".format(image_dir)))
if len(image_list) == 0:
    print("Warning: nothing found in directory")

# 2 run with real data
for input_filename in image_list:

    print(input_filename)

    # orig_img = fits.getdata(input_filename)

    im = np.array(Image.open(input_filename))
    x = transforms.ToTensor()(im).unsqueeze(0) # TODO maybe we should unsqueeze with numpy instead



    print("x data range (min,mean,max):", torch.min(x), torch.mean(x), torch.max(x))  # 0-1

    x_np = x.cpu().detach().numpy()
    print(x_np.shape)

    ## Inference:

    result = exec_net.infer({"input": x_np})
    print(result.keys())

    for output in list(exec_net.outputs.keys()):
        print("output named", output, "gives", result[output].shape)

    compressed = result["output"]
    print("compressed.shape", compressed.shape)

    # save array for decompression "on ground"
    out_fname = os.path.join(
                        output_dir,
                        "compressed_" +  os.path.basename(input_filename).replace(".jpg", ".npy")
                )
    np.save(out_fname, compressed)

    # can't plot - latent space too large
    # compressed_for_plot = np.transpose(compressed[0], (1, 2, 0))
    # plt.imshow(compressed_for_plot)
    # plt.savefig(img_fname, dpi=100)
    # plt.close()