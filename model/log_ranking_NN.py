import torch
import torch.nn as nn

import numpy as np
import os

from ._model import _Model
from dataloader import ParticipationRankingLoader

from utils.pools import *

class LogRankingNN(_Model):
    def __init__(self):
        super().__init__()
        self.acc_threshold = 0.001
        self.model = nn.Sequential(
            nn.Linear(self.dataloader.input_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16,1),
        )

    def __repr__(self):
        return "Ranking Feedforward Neural Network"

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _dataloader():
        return ParticipationRankingLoader()

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    def accuracy(self, output, target):
        return (torch.round(torch.exp(output)) == torch.round(torch.exp(target))).float().mean().item()

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

    def process_y(self, y):
        return (np.log(y)).astype(np.float32).reshape(-1, 1)

    def display_results(self, **kwargs):
        pass

    def format_predictions_for_race(self, combinations, predictions):
        predictions = predictions.squeeze(1).tolist()
        corresponding = list(zip(combinations, predictions))
        winner_horse = min(corresponding, key=lambda x: x[1])

        return {
            WIN: winner_horse,
            "ALL": dict(corresponding),
        }
