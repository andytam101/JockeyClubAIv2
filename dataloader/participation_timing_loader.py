import numpy as np

from .loader import Loader
from .utils import *
from .independent_loader import *

from database import Participation, Race
import utils.utils as utils

from tqdm import tqdm

RACE_FEATURES = 9


class ParticipationTimingLoader(Loader):
    def __init__(self):
        super().__init__()

    @property
    def input_features(self):
        return INDEPENDENT_FEATURES * 2 + RACE_FEATURES

    @staticmethod
    def convert_race_to_dict(race):
        return {
            "id": race.id,
            "date": race.date,
            "race_class": race.race_class,
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
        race_class = convert_race_class(race["race_class"])
        distance = race["distance"]
        location = race["location"] == "Sha Tin"   # (i.e. 0 for Happy Valley, 1 for Sha Tin)
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
        ps = get_training_participations(session, start_date, end_date)
        m = len(ps)

        result_x = np.zeros((m, self.input_features), dtype=np.float32)
        result_y = np.zeros((m, 1), dtype=np.float32)
        for idx in tqdm(range(m), desc="Loading data"):
            p = ps[idx]
            opponent_count = get_number_of_participants(p.race) - 1
            opponents = np.zeros((opponent_count, INDEPENDENT_FEATURES), dtype=np.float32)

            result_x[idx, :RACE_FEATURES] = self.load_race_features(self.convert_race_to_dict(p.race))
            result_x[idx, RACE_FEATURES:INDEPENDENT_FEATURES + RACE_FEATURES] = load_one_independent_participation(p)

            flag = 0
            for i, opponent in enumerate(get_all_participants(p.race)):
                if opponent == p:
                    flag = 1
                    continue
                opponents[i - flag] = load_one_independent_participation(opponent)

            result_x[idx, INDEPENDENT_FEATURES + RACE_FEATURES:] = np.mean(opponents, axis=0)
            result_y[idx, 0] = time_to_number_of_seconds(p.finish_time)

        assert not np.isnan(result_y).any()
        return np.nan_to_num(result_x), result_y

    def load_predict(self, session, data):
        m = len(data)
        numbers = []
        result = np.zeros((m, self.input_features), dtype=np.float32)
        race_features = self.load_race_features(data)

        for idx in range(m):
            p_entry = data[idx]


        return numbers, result
