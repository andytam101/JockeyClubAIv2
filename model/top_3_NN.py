import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os

from model._model import _Model
from dataloader.simple_loader import SimpleLoader

class Top3NN(_Model):
    """
    Top 3 single class classification. Feedforward Neural Network.
    Input vector: TODO: fill in input vector in comments
    """
    def __init__(self):
        super(Top3NN, self).__init__()
        self.fc1 = nn.Linear(self._dataloader().input_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def __repr__(self):
        return "Top 3 Feedforward Neural Network"

    @staticmethod
    def _dataloader():
        return SimpleLoader()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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