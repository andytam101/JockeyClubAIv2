import torch

from .winner_binary import WinnerBinary
from .place_binary import PlaceBinary
from .timing import Timing
from .win_odds import WinOdds
from .ranking_score import RankingScore

import utils.config as config


def load_model(model_name, model_config):
    models_mapping = {
        "WinBin": WinnerBinary,
        "PlaceBin": PlaceBinary,
        "Timing": Timing,
        "WinOdds": WinOdds,
        "RScore": RankingScore,
    }

    input_size = model_config["input_features"]
    return models_mapping[model_name](input_size=input_size).to(config.device)
