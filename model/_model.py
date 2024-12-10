from torch import nn
from abc import ABC, abstractmethod


class _Model(nn.Module, ABC):
    def __init__(self):
        super(_Model, self).__init__()

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
