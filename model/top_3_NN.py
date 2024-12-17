import torch
import torch.nn as nn
import torch.optim as optim

import os

from model._model import _Model


class Top3NN(_Model):
    """
    Top 3 single class classification. Feedforward Neural Network.
    Input vector: TODO: fill in input vector in comments
    """
    def __init__(self, dataloader, **kwargs):
        super(Top3NN, self).__init__(dataloader)
        self.fc1 = nn.Linear(kwargs["input_dim"], 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

        self.model_state_dict_path = None
        self.optimizer_state_dict_path = None
        self.train_mean_path = None
        self.train_std_path = None

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.output(x))

    def optimizer(self):
        optimiser = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if self.optimizer_state_dict_path is not None:
            optimiser.load_state_dict(torch.load(self.optimizer_state_dict_path))
        return optimiser

    @staticmethod
    def criterion():
        return nn.BCELoss()

    def __repr__(self):
        return "Top 3 Feedforward Neural Network"

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            output = self.forward(x)
            return output

    def accuracy(self, output, target):
        return (torch.round(output) == target).float().mean().item()

    def load(self, model_dir):
        super().load(model_dir)

        self.model_state_dict_path = os.path.join(model_dir, "model_state.pth")
        self.optimizer_state_dict_path = os.path.join(model_dir, "optimizer_state.pth")
        self.train_mean_path = os.path.join(model_dir, "train_mean.npy")
        self.train_std_path  = os.path.join(model_dir, "train_std.npy")

        self.load_state_dict(torch.load(self.model_state_dict_path))