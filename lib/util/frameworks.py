import logging

import torch

logger = logging.getLogger(__name__)


def getDevice(force_cpu=False):
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    logger.info(f'Using {device} device. (Notice: {torch.cuda.device_count()} cuda device available.)')
    return device


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    getDevice()
