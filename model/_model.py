import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod


class _Model(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def criterion(self):
        raise NotImplementedError

    def format_y(self, y):
        return y

    @abstractmethod
    def perform_bet(self, horse_nums, x, **kwargs):
        """
        Only used for evaluating model. Will not be used for actual prediction.
        """
        raise NotImplementedError
