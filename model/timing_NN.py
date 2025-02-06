import torch
import torch.nn as nn

from dataloader import PointwiseLoader
from ._model import _Model

from utils.pools import *


class TimingNN(_Model):
    def __init__(self):
        super().__init__()
        self.accuracy_threshold = 0.1

        self.model = nn.Sequential(
            nn.Linear(self.dataloader.input_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _dataloader():
        return PointwiseLoader()

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=1e-5, weight_decay=0.01, momentum=0.8)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    def accuracy(self, output, target):
        accuracy_threshold = self.accuracy_threshold
        return ((target - accuracy_threshold < output) & (output < target + accuracy_threshold)).float().mean().item()

    def process_y(self, y):
        return y[:, 2].reshape(-1, 1)

    def display_results(self, **kwargs):
        raise NotImplementedError

    def format_predictions_for_race(self, combinations, predictions):
        combinations = list(map(int, combinations))
        predictions = predictions.tolist()

        corresponding = list(zip(combinations, predictions))
        corresponding.sort(key=lambda x: x[1])

        first = corresponding[0]

        return {
            WIN: first
        }