import yaml
from easydict import EasyDict


class Config:
    def __init__(self):
        self.data = None

    def load(self, path):
        with open(path) as file:
            self.data = EasyDict(yaml.safe_load(file))

    def get_hyper(self):
        if self.data:
            return self.data
        raise Exception('Config file has not been loaded yet.')


config = Config()
