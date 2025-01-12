import torch

from model.model_prediction import ModelPrediction
from strategy._Strategy import Strategy
from model import load_model

from utils.config import device


class HighPWinnerProportionStrategy(Strategy):
    def __init__(self, model_dir, **kwargs):
        super().__init__()
        model = load_model("WinnerNN")
        self.predictor = ModelPrediction(model, model_dir)
        self.threshold = kwargs["threshold"]
        self.count = kwargs["count"]

    def __repr__(self):
        return f"Highest Probability (c={self.count}) (t={self.threshold}) Winner Proportional Strategy"

    def _bet(self, session, data):
        _, prediction = self.predictor.predict(session, data)
        prediction = prediction.squeeze(1)
        horse_nums = list(map(lambda x: x["number"], data))
        win_odds = torch.tensor(list(map(lambda x: x["win_odds"], data)), dtype=torch.float32, device=device)

        expected = prediction * win_odds
        prediction = prediction.tolist()
        expected = expected.tolist()

        corresponding = list(zip(horse_nums, prediction, expected))
        corresponding.sort(key=lambda x: x[1], reverse=True)

        result = {}
        for i in range(self.count):
            winner = corresponding[i]
            result[winner[0]] = self.balance * 0.1 if winner[2] > self.threshold else 0

        return {"win": result}
