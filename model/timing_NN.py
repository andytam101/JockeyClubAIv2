import torch
import torch.nn as nn

from dataloader import ParticipationTimingLoader
from ._model import _Model

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
        return ParticipationTimingLoader()

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
        return y

    def display_results(self, **kwargs):
        raise NotImplementedError

    def format_predictions_for_race(self, combinations, predictions):
        raise NotImplementedError
