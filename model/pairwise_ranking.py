import torch
import torch.nn as nn

import numpy as np

from dataloader import PairwiseLoader
from ._model import _Model

from utils.pools import *


class PairwiseRanking(_Model):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.dataloader.input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _dataloader():
        return PairwiseLoader()

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.01, momentum=0.1)

    @staticmethod
    def criterion():
        return nn.L1Loss()

    def accuracy(self, output, target):
        return (torch.round(output * 13) == (target * 13)).float().mean().item()

    def process_y(self, y):
        return ((y[:, 1] - y[:, 0]) / 13).astype(np.float32).reshape(-1, 1)

    def display_results(self, **kwargs):
        pass

    def format_predictions_for_race(self, combinations, predictions):
        n1 = max(combinations, key=lambda x: x[1])[1]
        n2 = max(combinations, key=lambda x: x[0])[0]
        n = max(n1, n2)

        matrix = torch.zeros((n, n), dtype=torch.float64)
        for (i, j), diff in zip(combinations, predictions):
            matrix[i - 1, j - 1] = -diff
            matrix[j - 1, i - 1] = diff

        matrix.fill_diagonal_(1.0)
        probabilities = matrix.sum(dim=1)
        probabilities = torch.softmax(probabilities, dim=0)

        assert not torch.isnan(probabilities).any()
        new_combinations = list(range(1, n + 1))

        assert len(new_combinations) == len(probabilities.tolist())
        corresponding = list(zip(new_combinations, probabilities.tolist()))
        corresponding.sort(key=lambda x: x[1])

        first = corresponding[0]
        second = corresponding[1]
        third = corresponding[2]
        fourth = corresponding[3]

        return {
            WIN: first,
            PLACE: [first[0], second[0], third[0]],
            FORECAST: (first[0], second[0],),
            QUINELLA: (first[0], second[0],),
            Q_PLACE: [(first[0], second[0]), (first[0], third[0]), (second[0], third[0])],
            TRIO: (first[0], second[0], third[0]),
            TIERCE: (first[0], second[0], third[0]),
            FIRST_4: (first[0], second[0], third[0], fourth[0]),
            QUARTET: (first[0], second[0], third[0], fourth[0]),

            "ALL": dict(corresponding),
        }

