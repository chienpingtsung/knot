import logging

import torch.cuda

logger = logging.getLogger(__name__)


def get_device():
    logger.info(f'{torch.cuda.device_count()} cuda device(s) available.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using {device} device(s).')
    return device


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    get_device()
