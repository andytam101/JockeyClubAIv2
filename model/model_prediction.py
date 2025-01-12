from abc import abstractmethod

import torch

from ._model import _Model

import os
from utils.config import device

class ModelPrediction:
    def __init__(self, model, model_dir):
        self.model: _Model = model
        self.model_dir = model_dir
        self.dataloader = self.model.dataloader
        self.model.to(device)

        self.load()

    def load(self):
        model_state_dict_path = os.path.join(self.model_dir, 'model_state_dict.pth')
        model_state_dict = torch.load(model_state_dict_path, map_location=device)
        self.model.load_state_dict(model_state_dict)

    def predict(self, session, data):
        combinations, x = self.dataloader.load_predict(session, data)
        params = self.model.load_normalization(self.model_dir)
        self.dataloader.normalize(x, **params)
        x = torch.tensor(x, dtype=torch.float32, device=device)

        self.model.eval()
        predictions = self.model(x)
        formatted_predictions = self.model.reformat_predictions(predictions)
        return combinations, formatted_predictions

    def guess_outcome_of_race(self, session, data):
        combinations, predictions = self.predict(session, data)
        return self.model.format_predictions_for_race(combinations, predictions)

    def display_results(self, **kwargs):
        self.model.display_results(**kwargs)

    def __call__(self, session, data):
        return self.predict(session, data)
