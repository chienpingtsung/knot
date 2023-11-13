import argparse


def train_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config')
    parser.add_argument('--log_dir')
    parser.add_argument('--weights')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--scout', type=int)

    return parser.parse_args()
