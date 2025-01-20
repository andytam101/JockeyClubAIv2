from model.log_ranking_NN import LogRankingNN
from model.pairwise_binary import PairwiseBinary
from model.pairwise_ranking import PairwiseRanking
from model.ranking_NN import RankingNN
from model.timing_NN import TimingNN
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
    "PairBinary": PairwiseBinary,
    "PairRanking": PairwiseRanking,
    "TimingNN": TimingNN,
}


def load_model(model_name):
    model = model_dict[model_name]
    return model().to(config.device)
