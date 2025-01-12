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


class Top3NN(_Model):
    """
    Top 3 single class classification. Feedforward Neural Network.
    Input vector: TODO: fill in input vector in comments
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.dataloader.input_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def __repr__(self):
        return "Top 3 Neural Network"

    @staticmethod
    def _dataloader():
        return ParticipationRankingLoader()

    def optimizer(self):
        return optim.Adam(self.parameters(), lr=0.003)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.bn1(x)
        # x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        # x = self.bn2(x)
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
        return (y <= 3).astype(np.float32).reshape(-1, 1)

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

    def format_predictions_for_race(self, combinations, predictions):
        predictions = predictions.squeeze(1).tolist()
        corresponding = list(zip(combinations, predictions))
        corresponding.sort(key=lambda x: x[1], reverse=True)
        top_3 = list(map(lambda x: x[0], corresponding[:3]))

        return {
            PLACE: top_3
        }

    def display_results(self, **kwargs):
        chi_names = kwargs["chi_names"]
        results = kwargs["results"]
        win_odds = list(map(lambda x: x[1], kwargs["win_odds"]))

        headers = ["名字", "賠率", "位置機會率", "值博率"]
        results_list = results.squeeze(1).tolist()
        multiplied = list(map(lambda x: x[0] * x[1], zip(results_list, win_odds)))

        desired_width = max(wcswidth(chi_str) for chi_str in chi_names + headers)

        results_list = [f"{(pred * 100):.1f}%" for pred in results_list]
        win_odds = [f"{odd:.2f}" for odd in win_odds]
        multiplied = [f"{val:.3f}" for val in multiplied]
        chi_names = [pad_chinese(name, desired_width) for name in chi_names]
        res = list(map(list, zip(chi_names, win_odds, results_list, multiplied)))
        res.sort(key=lambda x: x[3], reverse=True)

        print(tabulate(
            res,
            headers=headers,
            tablefmt="psql",
            colalign=["center"] * len(res[0]),
            floatfmt=".3f"
        ))
