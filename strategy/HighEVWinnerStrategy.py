import torch

from model.model_prediction import ModelPrediction
from strategy._Strategy import Strategy
from model import load_model

from utils.config import device

class HighEVWinnerStrategy(Strategy):
    def __init__(self, model_dir, **kwargs):
        super().__init__()
        model = load_model("WinnerNN")
        self.predictor = ModelPrediction(model, model_dir)

    def __repr__(self):
        return "Highest Expected Value Winner Strategy"

    def bet(self, session, data):
        prediction = self.predictor.predict(session, data).squeeze(1)
        horse_nums = list(map(lambda x: x["number"], data))
        win_odds = torch.tensor(list(map(lambda x: x["win_odds"], data)), dtype=torch.float32, device=device)

        expected = prediction * win_odds
        expected = expected.tolist()
        corresponding = list(zip(horse_nums, expected))
        winner = max(corresponding, key=lambda x: x[1])
        return {
            "win": {
                winner[0]: 10,
            }
        }
