import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
from tabulate import tabulate
from wcwidth import wcswidth

from dataloader import ParticipationRankingLoader
from ._model import _Model
from utils.utils import pad_chinese
from utils.pools import *


class WinnerNN(_Model):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.dataloader.input_features, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout3 = nn.Dropout(0.2)

        self.output = nn.Linear(8, 1)

    def __repr__(self):
        return "Winner Feedforward Neural Network."

    @staticmethod
    def _dataloader():
        return ParticipationRankingLoader()

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=0.001, weight_decay=0.005)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        # x = self.dropout1(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        # x = self.dropout2(x)

        return torch.sigmoid(self.output(x))

    @staticmethod
    def criterion():
        return nn.BCELoss()

    def accuracy(self, output, target):
        return (torch.round(output) == target).float().mean().item()

    @staticmethod
    def reformat_predictions(predictions):
        return predictions

    def process_y(self, y):
        return (y == 1).astype(np.float32).reshape(-1, 1)

    def format_predictions_for_race(self, combinations, predictions: torch.tensor):
        predictions = predictions.squeeze(1).tolist()
        corresponding = list(zip(combinations, predictions))
        winner_horse = max(corresponding, key=lambda x: x[1])

        return {
            WIN: winner_horse,
            "ALL": dict(corresponding),
        }

    def display_results(self, **kwargs):
        chi_names = kwargs["chi_names"]
        results = kwargs["results"]
        win_odds = list(map(lambda x: x[0], kwargs["win_odds"]))

        headers = ["名字", "賠率", "獨贏機會率", "值博率"]
        results_list = results.squeeze(1).tolist()
        multiplied = list(map(lambda x: x[0] * x[1], zip(results_list, win_odds)))

        desired_width = max(wcswidth(name) for name in chi_names)

        results_list = [f"{(pred * 100):.1f}%" for pred in results_list]
        win_odds = [f"{odd:.2f}" for odd in win_odds]
        multiplied = [f"{val:.3f}" for val in multiplied]
        chi_names = [pad_chinese(name, desired_width) for name in chi_names]
        res = list(map(list, zip(chi_names, win_odds, results_list, multiplied)))
        res.sort(key=lambda x: float(x[3]), reverse=True)

        print(tabulate(
            res,
            headers=headers,
            tablefmt="psql",
            colalign=["center"] * len(res[0]),
            floatfmt=".3f",
        ))
