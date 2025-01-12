import torch
from torch import nn
from abc import ABC, abstractmethod

from dataloader.loader import Loader as DataLoader
from utils.config import device


class _Model(nn.Module, ABC):
    def __init__(self):
        super(_Model, self).__init__()
        self.dataloader: DataLoader = self._dataloader()

    def predict(self, data):
        x = self.dataloader.load_predict(data)
        self.dataloader.normalize(x, **self.normalization)
        x = torch.tensor(x, device=device)
        predictions = self(x)
        return self.reformat_predictions(predictions)

    @staticmethod
    @abstractmethod
    def _dataloader():
        """
        Returns basic unloaded type of dataloader required by each model
        """
        raise NotImplementedError

    @abstractmethod
    def optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def criterion():
        raise NotImplementedError

    @abstractmethod
    def accuracy(self, output, target):
        raise NotImplementedError

    @staticmethod
    def reformat_predictions(predictions):
        return predictions

    @abstractmethod
    def save_normalization(self, model_dir, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_normalization(self, model_dir):
        raise NotImplementedError

    @abstractmethod
    def process_y(self, y):
        raise NotImplementedError

    @abstractmethod
    def display_results(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def format_predictions_for_race(self, combinations, predictions):
        raise NotImplementedError