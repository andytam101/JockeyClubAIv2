import torch
import torch.nn as nn
import torch.optim as optim

from model._model import _Model


class Top3LR(_Model):
    """
    Top 3 single class classification. Logistic Regression.
    Input vector: TODO: fill in input vector in comments
    """
    def __init__(self, **kwargs):
        super(Top3LR, self).__init__()
        self.linear = nn.Linear(kwargs["input_dim"], 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    @staticmethod
    def criterion():
        return nn.BCELoss()

    def __repr__(self):
        return "Top 3 Logistic Regression"

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            output = self.forward(x)
            return output

    def accuracy(self, output, target):
        return (torch.round(output) == target).float().mean().item()
