import torch
import torch.nn as nn
import numpy as np

from dataloader import PointwiseLoader
from ._model import _Model

from utils.pools import *


class WinnerNN(_Model):
    def __init__(self):
        super().__init__()
        self.accuracy_threshold = 0.1

        self.model = nn.Sequential(
            nn.Linear(self.dataloader.input_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def _dataloader():
        return PointwiseLoader()

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4, weight_decay=0.001, momentum=0.8)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def criterion():
        return nn.BCELoss()

    def accuracy(self, output, target):
        accuracy_threshold = self.accuracy_threshold
        return ((target - accuracy_threshold < output) & (output < target + accuracy_threshold)).float().mean().item()

    def process_y(self, y):
        return (y[:, 0] == 1).astype(np.float32).reshape(-1, 1)

    def display_results(self, **kwargs):
        raise NotImplementedError

    def format_predictions_for_race(self, combinations, predictions):
        combinations = list(map(int, combinations))
        predictions = predictions.tolist()

        corresponding = list(zip(combinations, predictions))
        corresponding.sort(key=lambda x: x[1], reverse=True)

        first = corresponding[0]

        return {
            WIN: first
        }