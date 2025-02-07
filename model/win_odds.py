import torch
from torch import nn
import torch.optim as optim

from ._model import _Model


class WinOdds(_Model):
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
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    def criterion(self):
        return nn.MSELoss()

    def format_y(self, y):
        return (y[:, 3]).unsqueeze(1).float()
