from torch import nn
from abc import ABC, abstractmethod


class _Model(nn.Module, ABC):
    def __init__(self, dataloader):
        super(_Model, self).__init__()
        self.from_load = False
        self.dataloader = dataloader

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def optimizer(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def criterion():
        raise NotImplementedError

    @abstractmethod
    def accuracy(self, output, target):
        raise NotImplementedError

    def load(self, model_dir):
        self.from_load = True
