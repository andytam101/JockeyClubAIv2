from database import Participation
from .loader import Loader
import numpy as np
from itertools import combinations

from .utils import *
from tqdm import tqdm

import random


class PairwiseLoader(Loader):
    @property
    def input_features(self):
        return INDIVIDUAL_FEATURES * 2

    def _load_from_db(self, session, start_date=None, end_date=None):
        races = session.query(Race)
        if start_date is not None:
            races = races.filter(Race.date >= start_date)
        if end_date is not None:
            races = races.filter(Race.date < end_date)

        races = races.all()
        m = 0
        for race in races:
            num_ps = len(utils.remove_unranked_participants(race.participations))
            m += (num_ps * (num_ps - 1) // 2)

        data_x = np.zeros((m, self.input_features), dtype=np.float32)
        data_y = np.zeros((m, 2), dtype=np.float32)
        counter = 0

        for race in tqdm(races, desc="Loading data"):
            ps = utils.remove_unranked_participants(race.participations)
            for i in range(len(ps) - 1):
                for j in range(i + 1, len(ps)):
                    if random.random() < 0.5:
                        first_p = ps[i]
                        snd_p = ps[j]
                    else:
                        first_p = ps[j]
                        snd_p = ps[i]

                    data_x[counter, :INDIVIDUAL_FEATURES] = load_individual_participation(session, first_p)
                    data_x[counter, INDIVIDUAL_FEATURES:] = load_individual_participation(session, snd_p)

                    data_y[counter, 0] = get_ranking_from_participation(first_p)
                    data_y[counter, 1] = get_ranking_from_participation(snd_p)

                    counter += 1
        return data_x, data_y


    def load_predict(self, session, data):
        n = len(data)
        size = n * (n - 1) // 2
        pairings = []
        result = np.zeros((size, self.input_features), dtype=np.float32)
        counter = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    p1 = data[i]
                    p2 = data[j]
                else:
                    p1 = data[j]
                    p2 = data[i]

                result[counter, :INDIVIDUAL_FEATURES] = load_individual_predict(session, **p1)
                result[counter, INDIVIDUAL_FEATURES:] = load_individual_predict(session, **p2)

                counter += 1
                pairings.append((p1["number"], p2["number"]))
        return pairings, result
