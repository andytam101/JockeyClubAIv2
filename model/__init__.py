from model.log_ranking_NN import LogRankingNN
from model.ranking_NN import RankingNN
from model.top_3_NN import Top3NN
from model.top_3_LR import Top3LR
from model.winner_NN import WinnerNN
from utils import config

model_dict = {
    "Top3LR": Top3LR,
    "Top3NN": Top3NN,
    "WinnerNN": WinnerNN,
    "RankingNN": RankingNN,
    "LRankingNN": LogRankingNN,
}


def load_model(model_name):
    model = model_dict[model_name]
    return model().to(config.device)
