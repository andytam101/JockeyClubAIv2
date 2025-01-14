import torch
from torch import nn
from abc import ABC, abstractmethod

from dataloader.loader import Loader as DataLoader
from utils.config import device

import numpy as np
import os


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

    def save_normalization(self, model_dir, **kwargs):
        mean_path = os.path.join(model_dir, "train_mean.npy")
        std_path = os.path.join(model_dir, "train_std.npy")

        np.save(mean_path, kwargs["train_mean"])
        np.save(std_path, kwargs["train_std"])

    def load_normalization(self, model_dir):
        mean_path = os.path.join(model_dir, "train_mean.npy")
        std_path = os.path.join(model_dir, "train_std.npy")

        train_mean = np.load(mean_path)
        train_std = np.load(std_path)
        return {
            "train_mean": train_mean,
            "train_std": train_std
        }

    @abstractmethod
    def process_y(self, y):
        raise NotImplementedError

    @abstractmethod
    def display_results(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def format_predictions_for_race(self, combinations, predictions):
        raise NotImplementedError