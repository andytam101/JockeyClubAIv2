import torch

from model.model_prediction import ModelPrediction
from ._Strategy import Strategy
from model import load_model
from bet import Bet

from utils.pools import *


class PairwiseStrategy(Strategy):
    def __init__(self, model_dir, **kwargs):
        super().__init__(**kwargs)
        self.model = load_model("PairBinary")
        self.predictor = ModelPrediction(self.model, model_dir)

    def __repr__(self):
        return "Pairwise Comparison Strategy"

    def _bet(self, session, data):
        combinations, predictions = self.predictor.predict(session, data)

        n1 = max(combinations, key=lambda x: x[1])[1]
        n2 = max(combinations, key=lambda x: x[0])[0]
        n = max(n1, n2)

        matrix = torch.zeros((n, n), dtype=torch.float64)
        for (i, j), prob in zip(combinations, predictions):
            matrix[i - 1, j - 1] = 1 - prob
            matrix[j - 1, i - 1] = prob

        matrix.fill_diagonal_(1.0)
        probabilities = matrix.prod(dim=1)
        probabilities = torch.softmax(probabilities, dim=0)

        assert not torch.isnan(probabilities).any()
        new_combinations = list(range(1, n + 1))

        assert len(new_combinations) == len(probabilities.tolist())
        corresponding = list(zip(new_combinations, probabilities.tolist()))
        corresponding.sort(key=lambda x: x[1])

        first = corresponding[0]
        second = corresponding[1]
        third = corresponding[2]
        fourth = corresponding[3]

        return [
            Bet(combination=first[0], pool=WIN, amount=10, probability=first[1]),
            Bet(combination=(first[0], second[0]), pool=QUINELLA, amount=10, probability=first[1] * second[1] * 2),
            Bet(combination=(first[0], second[0]), pool=FORECAST, amount=10, probability=first[1] * second[1]),
            Bet(combination=first[0], pool=PLACE, amount=10, probability=first[1] * 3),
            Bet(combination=second[0], pool=PLACE, amount=10, probability=second[1] * 3),
            Bet(combination=third[0], pool=PLACE, amount=10, probability=third[1] * 3),
            Bet(combination=(first[0], second[0]), pool=Q_PLACE, amount=10, probability=first[1] * second[1] * 3),
            Bet(combination=(first[0], third[0]), pool=Q_PLACE, amount=10, probability=first[1] * third[1] * 3),
            Bet(combination=(second[0], third[0]), pool=Q_PLACE, amount=10, probability=second[1] * third[1] * 3),
            # Bet(combination=(first[0], second[0], third[0]), pool=TRIO, amount=10,
            #     probability=first[1] * second[1] * third[1] * 3),
            # Bet(combination=(first[0], second[0], third[0]), pool=TIERCE, amount=10,
            #     probability=first[1] * second[1] * third[1]),
            # Bet(combination=(first[0], second[0], third[0], fourth[0]), pool=FIRST_4, amount=10,
            #     probability=first[1] * second[1] * third[1] * fourth[1] * 6),
            # Bet(combination=(first[0], second[0], third[0], fourth[0]), pool=QUARTET, amount=10,
            #     probability=first[1] * second[1] * third[1] * fourth[1]),

        ]
