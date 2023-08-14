import torch

# “Variational Image Compression with a Scale Hyperprior”
from compressai.zoo import bmshj2018_factorized
device = 'cpu'

class compress(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.forward_func = model.g_a

  def forward(self, x):
    with torch.no_grad():
      y = self.forward_func(x)
    return y

net = bmshj2018_factorized(quality=2, pretrained=True).train().to(device)
compressor = compress(net)



# Input to the model
x = torch.randn(1, 3, 512, 512, requires_grad=True)

example_out = compressor(x)
print("example in:", x.shape)
try:
  print("example out:", example_out.shape)
except:
  print("example out:", example_out)


# Export the model - the full model "net"
torch.onnx.export(net,                       # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "onboard_net.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  verbose=True
)


# Export just the compression part of the model - "compressor"

torch.onnx.export(compressor,                       # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "onboard_compressor_y.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  verbose=True
)

# Save the bottleneck encoder separately
entropy_model_name = "saved_entropy_bottleneck.pt"
torch.save(net.entropy_bottleneck, entropy_model_name)


print("done!")
print("now run:")
print("")