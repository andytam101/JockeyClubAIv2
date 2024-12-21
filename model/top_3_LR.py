import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os

from model._model import _Model
from dataloader.simple_loader import SimpleLoader


class Top3LR(_Model):
    """
    Top 3 single class classification. Logistic Regression.
    Input vector: TODO: fill in input vector in comments
    """
    def __init__(self, output_dir, **kwargs):
        super(Top3LR, self).__init__(output_dir, kwargs.get("data_path"))
        self.linear = nn.Linear(19, 1)

        self.model_state_dict_path = None
        self.optimizer_state_dict_path = None
        self.train_mean_path = None
        self.train_std_path  = None

    def __repr__(self):
        return "Top 3 Logistic Regression"

    @staticmethod
    def _dataloader():
        return SimpleLoader()

    def _optimizer(self):
        return optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    @staticmethod
    def criterion():
        return nn.BCELoss()

    def accuracy(self, output, target):
        return (torch.round(output) == target).float().mean().item()

    @staticmethod
    def reformat_predictions(predictions):
        p_sum = torch.sum(predictions, dim=0)
        # multiply by 3 as there are 3 horses in Top 3 (i.e. sum of probabilities should be 300%)
        return (predictions / p_sum) * 3

    def _load_normalization(self, model_dir):
        mean_path = os.path.join(model_dir, "train_mean.npy")
        std_path  = os.path.join(model_dir, "train_std.npy")

        self.normalization = {
            "train_mean": np.load(mean_path),
            "train_std": np.load(std_path)
        }

    def _save_normalization(self, model_dir):
        mean_path = os.path.join(model_dir, "train_mean.npy")
        std_path  = os.path.join(model_dir, "train_std.npy")

        np.save(mean_path, self.normalization["train_mean"])
        np.save(std_path, self.normalization["train_std"])
