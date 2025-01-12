import torch

from model.model_prediction import ModelPrediction
from strategy._Strategy import Strategy
from model import load_model

from utils.config import device


class HighPWinnerProgressiveStrategy(Strategy):
    def __init__(self, model_dir, **kwargs):
        super().__init__()
        model = load_model("WinnerNN")
        self.predictor = ModelPrediction(model, model_dir)
        self.threshold = kwargs["threshold"]
        self.count = kwargs["count"]

        self.current_base = self.init_balance
        self.bet_count = 0

    def __repr__(self):
        return f"Highest Probability (c={self.count}) (t={self.threshold}) Winner Progressive Strategy"

    def update_base(self):
        if self.bet_count % 10 == 0:
            self.current_base = self.balance

    def _bet(self, session, data):
        combinations, prediction = self.predictor.predict(session, data)
        prediction = prediction.squeeze(1)
        win_odds = torch.tensor(list(map(lambda x: x["win_odds"], data)), dtype=torch.float32, device=device)

        expected = prediction * win_odds
        prediction = prediction.tolist()
        expected = expected.tolist()

        corresponding = list(zip(combinations, prediction, expected))
        corresponding.sort(key=lambda x: x[1], reverse=True)

        result = {}
        for i in range(self.count):
            winner = corresponding[i]
            if winner[2] > self.threshold:
                result[winner[0]] = self.current_base * 0.1
                self.bet_count += 1

        return {"win": result}

    def update_balance(self, profit):
        super().update_balance(profit)
        self.update_base()
