# Works with older 2021 version:
import openvino
import openvino.inference_engine
from openvino.inference_engine import IECore
ie = IECore()


def device_available():
    ie = IECore()
    devs = ie.available_devices
    return 'MYRIAD' in devs


