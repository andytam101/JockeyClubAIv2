import torch
import torch.nn as nn

from model._model import _Model


class Top3NN(_Model):
    """
    Top 3 single class classification. Feedforward Neural Network.
    Input vector: TODO: fill in input vector in comments
    """
    def __init__(self, input_dim=20):
        super(Top3NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.output(x))

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
