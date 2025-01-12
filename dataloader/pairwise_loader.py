from database import Participation
from .loader import Loader
import numpy as np
from itertools import combinations

from .utils import *
from tqdm import tqdm


class PairwiseLoader(Loader):
    @property
    def input_features(self):
        return INDIVIDUAL_FEATURES * 2

    def _load_from_db(self, session, start_date=None, end_date=None):
        ps = get_training_participations(session, start_date, end_date)
        n = len(ps)

        m = n * (n - 1) // 2

        result_x = np.zeros((m, self.input_features), dtype=np.float32)
        result_y = np.zeros((m, 2), dtype=np.float16)

        for first_idx in tqdm(range(n - 1), desc="Loading data"):
            for snd_idx in range(first_idx + 1, n):
                i = first_idx * n + snd_idx - first_idx
                result_x[i, :INDIVIDUAL_FEATURES] = load_individual_participation(session, ps[first_idx])
                result_x[i, INDIVIDUAL_FEATURES:] = load_individual_participation(session, ps[snd_idx])

                result_y[i, 0] = get_ranking_from_participation(ps[first_idx])
                result_y[i, 1] = get_ranking_from_participation(ps[snd_idx])

        return np.nan_to_num(result_x)

    def load_predict(self, session, data):
        pass
