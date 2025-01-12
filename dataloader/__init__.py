from dataloader.pairwise_loader import PairwiseLoader
from dataloader.participation_ranking_loader import ParticipationRankingLoader
import numpy as np

dataloader_dict = {
    "PRLoader": ParticipationRankingLoader,
    "PairLoader": PairwiseLoader
}


def load_dataloader(dl_name):
    dataloader = dataloader_dict[dl_name]
    return dataloader()
