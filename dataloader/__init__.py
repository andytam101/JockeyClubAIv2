from .pairwise_loader import PairwiseLoader
from .participation_ranking_loader import ParticipationRankingLoader
from .participation_timing_loader import ParticipationTimingLoader
import numpy as np

dataloader_dict = {
    "PRLoader": ParticipationRankingLoader,    # pointwise model
    "PairLoader": PairwiseLoader,              # pairwise model
    "PTLoader": ParticipationTimingLoader,     # pointwise model
}


def load_dataloader(dl_name):
    dataloader = dataloader_dict[dl_name]
    return dataloader()
