import numpy as np
import torch
from easydict import EasyDict
from torchvision.transforms import functional


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class ToTensor:
    def __call__(self, data):
        ret = EasyDict()
        for key, value in data.items():
            if isinstance(value, list):
                ret[key] = [functional.to_tensor(f) for f in value]
            elif isinstance(value, np.ndarray):
                ret[key] = torch.Tensor(value)
            else:
                ret[key] = value
        return ret


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data):
        ret = EasyDict()
        for key, value in data.items():
            if isinstance(value, list):
                ret[key] = [functional.normalize(f, self.mean, self.std, self.inplace) for f in value]
            else:
                ret[key] = value
        return ret
