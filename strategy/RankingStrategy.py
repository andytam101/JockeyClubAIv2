from model import load_model
from model.model_prediction import ModelPrediction
from strategy._Strategy import Strategy

from utils.pools import *
from bet import Bet


class RankingStrategy(Strategy):
    def __init__(self, model_dir, **kwargs):
        super().__init__(**kwargs)
        self.model = load_model("RankingNN")
        self.predictor = ModelPrediction(self.model, model_dir)

    def __repr__(self):
        return "Ranking Strategy"

    def _bet(self, session, data):
        combinations, ranking = self.predictor.predict(session, data)
        corresponding = list(zip(combinations, ranking))
        corresponding.sort(key=lambda x: x[1])

        first = corresponding[0]
        second = corresponding[1]
        third = corresponding[2]
        fourth = corresponding[3]

        return [
            Bet(combination=first[0], pool=WIN, amount=10, probability=None),
            # Bet(combination=(first[0], second[0]), pool=QUINELLA, amount=10, probability=None),
            Bet(combination=(first[0], second[0]), pool=FORECAST, amount=10, probability=None),
            Bet(combination=first[0], pool=PLACE, amount=10, probability=None),
            # Bet(combination=second[0], pool=PLACE, amount=10, probability=None),
            # Bet(combination=third[0], pool=PLACE, amount=10, probability=None),
            Bet(combination=(first[0], second[0]), pool=Q_PLACE, amount=10, probability=None),
            # Bet(combination=(first[0], third[0]), pool=Q_PLACE, amount=10, probability=None),
            # Bet(combination=(second[0], third[0]), pool=Q_PLACE, amount=10, probability=None),
        ]
