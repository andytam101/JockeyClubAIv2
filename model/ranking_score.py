import torch
import torch.nn as nn
import torch.optim as optim

from ._model import _Model
from .utils import convert_ranking_to_score

from utils.pools import *


class RankingScore(_Model):
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
        )

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def criterion(self):
        return nn.MSELoss()

    def format_y(self, y):
        return convert_ranking_to_score(y[:, 0]).unsqueeze(dim=1)

    def perform_bet(self, horse_nums, x, **kwargs):
        k = kwargs.get('k', 0)

        predictions = self.forward(x)
        corresponding = list(zip(horse_nums, predictions))
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
