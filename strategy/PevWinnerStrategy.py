import torch

from model.model_prediction import ModelPrediction
from strategy._Strategy import Strategy
from model import load_model

from utils.config import device


class PevWinnerStrategy(Strategy):
    def __init__(self, model_dir, **kwargs):
        super().__init__()
        model = load_model("WinnerNN")
        self.predictor = ModelPrediction(model, model_dir)
        self.w = kwargs["w"]

    def __repr__(self):
        return f"PEV (w={self.w}) Winner Strategy"

    def bet(self, session, data):
        w = self.w

        prediction = self.predictor.predict(session, data).squeeze(1)
        horse_nums = list(map(lambda x: x["number"], data))
        win_odds = torch.tensor(list(map(lambda x: x["win_odds"], data)), dtype=torch.float32, device=device)

        balanced = w * prediction + (1 - w) * (prediction * win_odds)
        balanced = balanced.tolist()
        corresponding = list(zip(horse_nums, balanced))
        winner = max(corresponding, key=lambda x: x[1])
        return {
            "win": {
                winner[0]: 10 if winner[1] > 1 else 0,
            }
        }
