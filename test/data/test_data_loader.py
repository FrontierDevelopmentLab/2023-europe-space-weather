import sunerf
from icarus.data.data_loader import NeRFDataModule
import argparse
import yaml

data =NeRFDataModule(config_data) 




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='../../config/hyperparams_icarus.yaml')
    parser.add_argument('--train', default='../../config/train.yaml', type=str)

    args = parser.parse_args()
    config_data = {}
    with open(args.config, 'r') as stream:
        config_data.update(yaml.load(stream, Loader=yaml.SafeLoader))

    with open(args.train, 'r') as stream:
        config_data(yaml.load(stream, Loader=yaml.SafeLoader))