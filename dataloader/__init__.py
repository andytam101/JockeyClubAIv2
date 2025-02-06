from .lambda_rank_loader import LambdaRankLoader
from .pairwise_loader import PairwiseLoader
from .pointwise_loader import PointwiseLoader
import numpy as np

dataloader_dict = {
    "PWLoader": PointwiseLoader,     # pointwise model
    "PairLoader": PairwiseLoader,    # pairwise model
    "LambdaRank": LambdaRankLoader
}


def load_dataloader(dl_name):
    dataloader = dataloader_dict[dl_name]
    return dataloader()
