import yaml
from easydict import EasyDict


class Config(EasyDict):
    def load(self, path):
        with open(path) as file:
            super().__init__(yaml.safe_load(file))


hyper = Config()
