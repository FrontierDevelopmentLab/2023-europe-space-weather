from pathlib import Path

# Works with older 2021 version:
import openvino
import openvino.inference_engine
from openvino.inference_engine import IECore

ie = IECore()

import glob
import logging
import sys
import time
from typing import Callable

import numpy as np
import torch

# import matplotlib.pyplot as plt
from astropy.io import fits
from torchvision import transforms

input_filename = "/payload/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts"
model_path = "/payload/onboard_compressor_y.onnx"
save_folder = "/results/"

entropy_model_name = "/payload/saved_entropy_bottleneck.pt"
loaded_entropy_bottleneck = torch.load(entropy_model_name)
print(loaded_entropy_bottleneck)


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

example_input = np.random.rand(BATCH_SIZE, 3, 512, 512)


device = "CPU"
ie = IECore()
model = OpenVinoModel(ie, model_path)
exec_net = ie.load_network(
    network=model.net, device_name=device, config=None, num_requests=1
)

result = exec_net.infer({"input": example_input})

for output in list(exec_net.outputs.keys()):
    print("output named", output, "gives", result[output].shape)


# 2 run with real data

orig_img = fits.getdata(input_filename)
print(orig_img.shape, orig_img.dtype)
print(np.min(orig_img), np.mean(orig_img), np.max(orig_img))
# plt.imshow(orig_img, cmap="gray")

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
x = x.unsqueeze(0).to("cpu")
print("x data range (min,mean,max):", torch.min(x), torch.mean(x), torch.max(x))  # 0-1

x_np = x.cpu().detach().numpy()
print(x_np.shape)

## Inference:

result = exec_net.infer({"input": x_np})

for output in list(exec_net.outputs.keys()):
    print("output named", output, "gives", result[output].shape)
compressed_y = result["output"]
print(
    "compressed_y.shape", compressed_y.shape
)  # should be torch.Size([1, 192, 32, 32])

# now use the entropy_bottleneck on cpu
compressed_strings = loaded_entropy_bottleneck.compress(torch.from_numpy(compressed_y))
print("compressed_strings", len(compressed_strings))  # should be 1
print(
    "compressed_strings[0]", len(compressed_strings[0])
)  # should be 5408 ~ got 4980 ?
# print("compressed_strings[1][0]", len(compressed_strings[1][0]))


strings = [compressed_strings]
shape = compressed_y.shape[-2:]
latent_name = save_folder + "latent_" + str(shape[0]) + "_" + str(shape[1])

# Save compressed forms:
with open(latent_name + ".bytes", "wb") as f:
    f.write(strings[0][0])


# 3 timing

num_images = 25
print("Timing", num_images, "inference passes")

time_onnx = 0
time_on_cpu = 0
for _ in range(num_images):
    start = time.perf_counter()
    result = exec_net.infer({"input": x_np})
    example_output = result["output"]
    mid = time.perf_counter()
    time_onnx += mid - start

    compressed_y = result["output"]
    compressed_strings = loaded_entropy_bottleneck(torch.from_numpy(compressed_y))

    end = time.perf_counter()
    time_on_cpu += end - mid


print(
    f"ONNX model in OpenVINO CPU Runtime/CPU: {time_onnx/num_images:.3f} with postprocessing on CPU: {time_on_cpu/num_images:.3f}"
    f"seconds per image, FPS: {num_images/(time_onnx+time_on_cpu):.2f} (added times together)"
)
