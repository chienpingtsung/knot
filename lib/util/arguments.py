import argparse

import torch


def parse_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config')
    parser.add_argument('--log_dir')
    parser.add_argument('--weights')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--scout', type=int)

    args = parser.parse_args()

    args.batch_size *= torch.cuda.device_count() if torch.cuda.is_available() else 1

    return args
