from .loader import Loader
from .utils import *
from .independent_loader import INDEPENDENT_FEATURES, load_one_independent_participation

from database import Race

from tqdm import tqdm

RACE_FEATURES = 9


class LambdaRankLoader(Loader):
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

    @property
    def input_features(self):
        return INDEPENDENT_FEATURES + RACE_FEATURES

    @property
    def output_features(self):
        return 1

    def _load_from_db(self, session, start_date=None, end_date=None):
        races = get_races_between_dates(session, start_date, end_date)
        n = len(races)

        all_x = {}
        all_y = {}

        for idx in tqdm(range(n), desc="Loading data"):
            race = races[idx]
            race_dict = self.convert_race_to_dict(race)
            ps = utils.remove_unranked_participants(race.participations)
            p_count = len(ps)
            if p_count <= 4:
                continue

            ps.sort(key=lambda x: x.number)

            this_x = np.zeros((p_count, self.input_features), dtype=np.float32)
            this_y = np.zeros((p_count, self.output_features), dtype=np.float32)

            for i, p in enumerate(ps):
                this_x[i, :RACE_FEATURES] = self.load_race_features(race_dict)
                this_x[i, RACE_FEATURES:RACE_FEATURES + INDEPENDENT_FEATURES] = load_one_independent_participation(p, session, False)

                this_y[i, 0] = get_ranking_from_participation(p)

            all_x[race.id] = this_x
            all_y[race.id] = this_y

        return all_x, all_y

    def load_predict(self, session, data):
        pass