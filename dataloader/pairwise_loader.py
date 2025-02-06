from database import Participation
from .loader import Loader
from .independent_loader import *
import numpy as np
from itertools import combinations

from .utils import *
from tqdm import tqdm

import random
RACE_FEATURES = 9


class PairwiseLoader(Loader):
    def __init__(self):
        super().__init__()

    @property
    def input_features(self):
        return INDEPENDENT_FEATURES * 2 + RACE_FEATURES

    @property
    def output_features(self):
        return 4

    @staticmethod
    def convert_race_to_dict(race):
        return {
            "id": race.id,
            "date": race.date,
            "race_class": convert_race_class(race.race_class),
            "distance": race.distance,
            "location": race.location,
            "course": race.course,
            "condition": race.condition,
            "total_bet": race.total_bet,
            "number_of_participants": get_number_of_participants(race),
        }

    @staticmethod
    def load_race_features(race):
        # race as a dictionary
        race_class = race["race_class"]
        distance = race["distance"]
        location = race["location"] == "Sha Tin"  # (i.e. 0 for Happy Valley, 1 for Sha Tin)
        width = utils.get_track_width(race["location"], race["course"])
        condition = race["condition"]
        total_bet = race["total_bet"]
        number_of_participants = race["number_of_participants"]
        try:
            race_upper_limit = utils.RACE_UPPER_LIMIT[int(race_class)]
            race_lower_limit = utils.RACE_LOWER_LIMIT[int(race_class)]
        except IndexError:
            race_upper_limit = 0
            race_lower_limit = 0
        return np.array([
            race_class,
            distance,
            location,
            encode_condition(condition),
            width,
            total_bet,
            number_of_participants,
            race_upper_limit,
            race_lower_limit,
        ], dtype=np.float32)

    def _load_from_db(self, session, start_date=None, end_date=None):
        races = get_races_between_dates(session, start_date, end_date)
        n = len(races)

        all_x = {}
        all_y = {}

        for idx in tqdm(range(n), desc="Loading data"):
            race = races[idx]
            ps = utils.remove_unranked_participants(race.participations)
            ps.sort(key=lambda x: x.number)
            race_dict = self.convert_race_to_dict(race)
            m = len(ps)
            number_of_pairs = m * (m - 1) // 2

            this_x = np.zeros((number_of_pairs, self.input_features), dtype=np.float32)
            this_y = np.zeros((number_of_pairs, self.output_features), dtype=np.float32)
            counter = 0

            for i in range(m - 1):
                for j in range(i + 1, m):
                    first_p = ps[i]
                    second_p = ps[j]

                    this_x[counter, :RACE_FEATURES] = self.load_race_features(race_dict)
                    this_x[counter, RACE_FEATURES: RACE_FEATURES + INDEPENDENT_FEATURES] = (
                        load_one_independent_participation(first_p, session, False))
                    this_x[counter, RACE_FEATURES + INDEPENDENT_FEATURES:] = (
                        load_one_independent_participation(second_p, session, False))

                    this_y[counter, 0] = get_ranking_from_participation(first_p)
                    this_y[counter, 1] = get_ranking_from_participation(second_p)
                    this_y[counter, 2] = time_to_number_of_seconds(first_p.finish_time)
                    this_y[counter, 3] = time_to_number_of_seconds(second_p.finish_time)
                    counter += 1

            all_x[race.id] = this_x
            all_y[race.id] = this_y

        return all_x, all_y

    def load_predict(self, session, data):
        raise NotImplementedError
        # n = len(data)
        # size = n * (n - 1) // 2
        # pairings = []
        # result = np.zeros((size, self.input_features), dtype=np.float32)
        # counter = 0
        # for i in range(n - 1):
        #     for j in range(i + 1, n):
        #         if random.random() < 0.5:
        #             p1 = data[i]
        #             p2 = data[j]
        #         else:
        #             p1 = data[j]
        #             p2 = data[i]
        #
        #         result[counter, :INDIVIDUAL_FEATURES] = load_individual_predict(session, **p1)
        #         result[counter, INDIVIDUAL_FEATURES:] = load_individual_predict(session, **p2)
        #
        #         counter += 1
        #         pairings.append((p1["number"], p2["number"]))
        # return pairings, result
