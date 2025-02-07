import torch
from torch import nn
import torch.optim as optim

from ._model import _Model
from utils.pools import *


class WinnerBinary(_Model):
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=1e-4, momentum=0.8, weight_decay=0.001)

    def criterion(self):
        return nn.BCELoss()

    def format_y(self, y, **kwargs):
        return (y[:, 0] == 1).unsqueeze(1).float()

    def perform_bet(self, horse_nums, x, **kwargs):
        k = kwargs.get("winner_k", 0)
        probabilities = self.forward(x).flatten().tolist()

        corresponding = list(zip(horse_nums, probabilities))
        corresponding.sort(key=lambda x: x[1], reverse=True)

        first = corresponding[0]
        second = corresponding[1]
        fourth = corresponding[3]
        if first[1] - second[1] > k:
            return [(WIN, first[0]), (PLACE, first[0])]
        elif first[1] - fourth[1] > k:
            return [(PLACE, first[0])]
        else:
            return []
