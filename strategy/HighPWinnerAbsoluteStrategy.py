import torch

from model.model_prediction import ModelPrediction
from strategy._Strategy import Strategy
from model import load_model

from utils.config import device


class HighPWinnerAbsoluteStrategy(Strategy):
    def __init__(self, model_dir, **kwargs):
        super().__init__()
        model = load_model("WinnerNN")
        self.predictor = ModelPrediction(model, model_dir)
        self.threshold = kwargs["threshold"]
        self.count = kwargs["count"]
        self.bet_amount = self.init_balance * 0.1

    def __repr__(self):
        return f"Highest Probability (c={self.count}) (t={self.threshold}) Winner Absolute Strategy"

    def _bet(self, session, data):
        combinations, prediction = self.predictor.predict(session, data)
        prediction = prediction.squeeze(1)
        win_odds = torch.tensor(list(map(lambda x: x["win_odds"], data)), dtype=torch.float32, device=device)

        expected = prediction * win_odds
        prediction = prediction.tolist()
        expected = expected.tolist()

        corresponding = list(zip(combinations, prediction, expected))
        # corresponding.sort(key=lambda x: x[1], reverse=True)
        maximum = max(corresponding, key=lambda x: x[1])

        if maximum[2] > self.threshold:
            return {
                "win": {
                    maximum[0]: 10,
                }
            }

        return {}
