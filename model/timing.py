import torch
from torch import nn
import torch.optim as optim

from ._model import _Model
from utils.pools import *


class Timing(_Model):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    def criterion(self):
        return nn.MSELoss()

    def format_y(self, y):
        min_timing = torch.min(y[:, 2])
        return (y[:, 2] - min_timing).unsqueeze(1).float()

    def perform_bet(self, horse_nums, x, **kwargs):
        k = kwargs.get('k', 0)

        predicted_timings = self.forward(x).flatten().tolist()
        corresponding = list(zip(horse_nums, predicted_timings))
        corresponding.sort(key=lambda x: x[1])

        first = corresponding[0]
        second = corresponding[1]
        fourth = corresponding[3]

        if second[1] - first[1] > k:
            return [(WIN, first[0]), (PLACE, first[0])]
        elif fourth[1] - first[1] > k:
            return [(PLACE, first[0])]
        else:
            return []
