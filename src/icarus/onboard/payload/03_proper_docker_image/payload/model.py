import logging
from openvino.inference_engine import IECore
ie = IECore()
import numpy as np
import time

class OpenVinoModel:
    def __init__(self, ie, model_path, batch_size=1):
        self.logger = logging.getLogger("model")
        print('\tModel path: {}'.format(model_path))
        self.net = ie.read_network(model_path, model_path[:-4] + '.bin')
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


# def load_model(model_path, device='MYRIAD'):
#     # Broadly copied from the OpenVINO Python examples
#     ie = IECore()
#     try:
#         ie.unregister_plugin('MYRIAD')
#     except:
#         pass
#
#     model = OpenVinoModel(ie, model_path)
#     tic = time.time()
#     print('Loading ONNX network to ',device,'...')
#
#     exec_net = ie.load_network(network=model.net, device_name=device,config=None, num_requests=1)
#     toc = time.time()
#     print('one, time elapsed : {} seconds'.format(toc - tic))
#
#     def predict(x: np.ndarray) -> np.ndarray:
#         """
#         Predict function using the myriad chip
#
#         Args:
#             x: (C, H, W) 3d tensor
#
#         Returns:
#             (C, H, W) 3D network logits
#
#         """
#         print(x.shape)
#
#         result = exec_net.infer({'input': x[np.newaxis]})
#         return result['output'][0]  # (n_class, H, W)
#
#     return predict