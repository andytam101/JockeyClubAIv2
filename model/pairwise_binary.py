import torch
import torch.nn as nn

import numpy as np

from dataloader import PairwiseLoader
from ._model import _Model

from utils.pools import *


class PairwiseBinary(_Model):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.dataloader.input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _dataloader():
        return PairwiseLoader()

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.03, weight_decay=0.001, momentum=0.9)

    @staticmethod
    def criterion():
        return nn.BCELoss()

    def accuracy(self, output, target):
        return (torch.round(output) == target).float().mean().item()

    def process_y(self, y):
        return ((y[:, 0] == y[:, 1]).astype(np.float32) * 0.5 + (y[:, 0] < y[:, 1]).astype(np.float32)).reshape(-1, 1)

    def display_results(self, **kwargs):
        pass

    def format_predictions_for_race(self, combinations, predictions):
        number_of_pairs = predictions.size(0)
        n = round((1 + np.sqrt(1 + 8 * number_of_pairs)) / 2)

        matrix = torch.zeros((n, n), dtype=torch.float64)
        counter = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                probability = predictions[counter]
                assert 0 <= probability <= 1
                matrix[i, j] = probability
                matrix[j, i] = 1 - probability
                counter += 1

        matrix.fill_diagonal_(1.0)
        probabilities = matrix.prod(dim=1)
        probabilities = probabilities / torch.sum(probabilities, dim=0)

        new_combinations = list(range(1, n + 1))

        assert len(new_combinations) == len(probabilities.tolist())
        corresponding = list(zip(new_combinations, probabilities.tolist()))
        corresponding.sort(key=lambda x: x[1], reverse=True)

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
        
