# Works with older 2021 version:
import openvino
import openvino.inference_engine
from openvino.inference_engine import IECore
ie = IECore()
import numpy as np
import time
import math
import torch

#from utils import device_available
from model import OpenVinoModel
from data import load_fits_as_np

def main(settings):
    print("=== RUN INFERENCE ===")
    print("settings:", settings)
    input_filename = args["input"]
    model_path = args["model"]
    entropy_model_name = args["model_ent"]
    save_folder = args["results_dir"]
    BATCH_SIZE = int(args["batch_size"])
    device = args["device"]
    num_images = int(args["num_of_images"])
    assert num_images % BATCH_SIZE == 0 # better do multiply of batch size
    resolution = int(args["resolution"])

    save_bottleneck = args["save_bottleneck"] == 'True'
    bottleneck_name = args["bottleneck_name"]


    loaded_entropy_bottleneck = torch.load(entropy_model_name)
    print(loaded_entropy_bottleneck)

    # 1 load real data
    x_np = load_fits_as_np(input_filename)

    # 2 load openvino model
    example_input = np.random.rand(BATCH_SIZE, 3, resolution, resolution)

    ie = IECore()
    model = OpenVinoModel(ie, model_path)
    model.set_batch_size(BATCH_SIZE)

    exec_net = ie.load_network(network=model.net, device_name=device, config=None, num_requests=1)
    result = exec_net.infer({'input': example_input})
    for output in list(exec_net.outputs.keys()):
      print("output named",output,"gives", result[output].shape)
    result = exec_net.infer({'input': x_np})

    for output in list(exec_net.outputs.keys()):
      print("output named",output,"gives", result[output].shape)


    compressed_y = result['output']
    print("compressed_y.shape", compressed_y.shape) # should be torch.Size([1, 192, 32, 32])

    # now use the entropy_bottleneck on cpu
    compressed_strings = loaded_entropy_bottleneck.compress( torch.from_numpy(compressed_y) )
    print("compressed_strings", len(compressed_strings)) # should be 1
    print("compressed_strings[0]", len(compressed_strings[0])) # should be 5408 ~ got 4980 ?
    # print("compressed_strings[1][0]", len(compressed_strings[1][0]))


    strings = [compressed_strings]
    shape = compressed_y.shape[-2:]

    if save_bottleneck:
        latent_name = save_folder + bottleneck_name + str(shape[0]) + "_" + str(shape[1])

        # Save compressed forms:
        with open(latent_name+".bytes", 'wb') as f:
            f.write(strings[0][0])


    # 3 timing
    print("Timing", num_images, "inference passes")

    time_onnx = 0
    time_on_cpu = 0

    num_batches = math.ceil(num_images / BATCH_SIZE)
    for batch_i in range(num_batches):
        image_start = batch_i * BATCH_SIZE
        image_end = (batch_i+1) * BATCH_SIZE - 1
        image_end = min(num_images, image_end)
        this_batch = image_end - image_start + 1
        print("batch #", batch_i, "images <", image_start, image_end, "> which has", this_batch)

        # Timed inference:
        # x->y in ONNX
        start = time.perf_counter()
        result = exec_net.infer({'input': x_np})
        compressed_y = result['output']

        mid = time.perf_counter()
        time_onnx += mid - start

        # y->strings in TORCH
        compressed_strings = loaded_entropy_bottleneck( torch.from_numpy(compressed_y) )

        end = time.perf_counter()
        time_on_cpu += end - mid
        print("- in:", x_np.shape, "out:", len(compressed_strings), "x", compressed_strings[0].shape)

    print("=== Results ===")
    print("DEVICE:", device)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("num_images:", num_images)
    print("total time_onnx:", time_onnx)
    print("total time_on_cpu:", time_on_cpu)
    print()
    print("In total: ", (time_onnx+time_on_cpu)/num_images, "per one image")
    print("FPS: ", num_images / (time_onnx+time_on_cpu), " (or images per second)")
    print()


if __name__ == "__main__":
    import argparse

    custom_path = ""
    # custom_path = "../" # only on test machine ...

    parser = argparse.ArgumentParser('Run inference')
    parser.add_argument('--input', default=custom_path+"data/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts",
                        help="Full path to file or folder of files in FITS format")
    parser.add_argument('--model', default=custom_path+'weights/onboard_compressor_y.onnx',
                        help="Path to the model weights")
    parser.add_argument('--model_ent', default=custom_path+'weights/saved_entropy_bottleneck.pt',
                        help="Path to the entropy encoder model weights")
    parser.add_argument('--results_dir', default=custom_path+'results/',
                        help="Path where to save the results")
    # parser.add_argument('--log_name', default='log',
    #                     help="Name of the log (batch size will be appended in any case).")
    parser.add_argument('--batch_size', default="1",
                        help="Batch size for the dataloader and inference")
    parser.add_argument('--device', default="CPU",
                        help="Device (choose either CPU or MYRIAD)")
    parser.add_argument('--num_of_images', default="4",
                        help="How many images to infer? (choose a multiple of the batch size)")
    parser.add_argument('--resolution', default="512",
                        help="Resolution of the image (for demo inference, must match the model)")

    parser.add_argument('--save_bottleneck', default="True",
                        help="Should we save the bottleneck (for later reconstruction outside)?")
    parser.add_argument('--bottleneck_name', default="latent_",
                        help="Name of the saved bottleneck")


    args = vars(parser.parse_args())
    main(settings=args)