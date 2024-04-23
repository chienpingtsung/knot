import argparse

import torch.cuda
import yaml
from easydict import EasyDict


class Hyper(EasyDict):
    def load(self, path):
        with open(path) as file:
            super().__init__(yaml.safe_load(file))


hyper = Hyper()


def parse_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hyper')
    parser.add_argument('--log_dir')
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--scout', type=int)

    args = parser.parse_args()

    args.batch_size *= torch.cuda.device_count() if torch.cuda.is_available() else 1

    return args
