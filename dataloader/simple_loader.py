from datetime import timedelta

import numpy as np

from database import Participation, Horse
from dataloader.loader import Loader
from dataloader.utils import *

from tqdm import tqdm


class SimpleLoader(Loader):
    def __init__(self, cv_percentage, save_dir=None):
        super().__init__(save_dir)
        self.cv_percentage = cv_percentage

        self.zscore_indices = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18]

    def normalise(self, data, training_mean=None, training_std=None):
        m = len(data)
        cv_index = int(m - m * self.cv_percentage)
        training_data = data[:cv_index]
        if training_mean is None or training_std is None:
            training_mean = np.mean(training_data, axis=0)
            training_std = np.std(training_data, axis=0)

        training_std[training_std == 0] = 1

        for i in range(data.shape[1]):
            if i in self.zscore_indices:
                # Z-score normalization
                data[:, i] = (data[:, i] - training_mean[i]) / training_std[i]
            else:
                # Leave as-is (if any feature doesn't require normalization, specify its indices here)
                pass

        return data, training_mean, training_std

    def load(self):
        # TODO: refactor this function to take in m (if None then set to length of ps)
        ps = self.remove_unranked_participants(self.session.query(Participation).all())
        m = len(ps)
        result_x = np.zeros((m, 19), dtype=np.float32)
        result_y = np.array(list(map(get_ranking_from_participation, ps[:m])), dtype=np.float32)
        result_y = (result_y <= 3).astype(np.float32)

        for idx, p in tqdm(enumerate(ps[:m]), desc="Loading data"):
            res = self.load_one_participation(p)
            result_x[idx, :] = np.copy(res)

        data_x, training_mean, training_std = self.normalise(result_x)
        self.save(data_x, result_y)
        return data_x, result_y, training_mean, training_std

    def load_n_participations(self, ps: list[Participation], training_mean, training_std):
        m = len(ps)
        result = np.zeros((m, 19), dtype=np.float32)
        for idx, p in enumerate(ps):
            result[idx] = self.load_one_participation(p)
        res = self.normalise(result, training_mean, training_std)
        return res

    def load_one_participation(self, p: Participation):
        static_data = (self.get_participation_data(p)
                       + self.get_race_data(p.race)
        )

        before = p.race.date
        after = before - timedelta(days=90)

        relevant_ps = (self.session.query(Participation).join(Race).join(Horse)
                       .filter(Race.date < before)
                       .filter(Race.date >= after)
                       )

        horse_ps = relevant_ps.filter(Participation.horse_id == p.horse_id).all()
        jockey_ps = relevant_ps.filter(Participation.jockey_id == p.jockey_id).all()
        trainer_ps = relevant_ps.filter(Horse.trainer_id == p.horse.trainer_id).all()

        result = np.array(static_data
                          + self.get_grouped_stats(self.remove_unranked_participants(horse_ps))
                          + self.get_grouped_stats(self.remove_unranked_participants(jockey_ps))
                          + self.get_grouped_stats(self.remove_unranked_participants(trainer_ps)), dtype=np.float32)

        result = np.nan_to_num(result)
        return result

    def remove_unranked_participants(self, ps):
        return list(filter(lambda x: x.ranking.replace("DH", "").strip().isnumeric(), ps))

    def get_participation_data(self, p: Participation):
        return [
            p.lane,
            p.gear_weight,
            p.horse_weight,
            p.win_odds
        ]

    def get_horse_data(self, h: Horse):
        return [
            h.age  # TODO: subtract now from race date
        ]

    def get_race_data(self, r: Race):
        return [
            convert_race_class(r.race_class),
            r.distance,
            r.total_bet
        ]

    def get_grouped_stats(self, ps: list[Participation]):
        speeds = list(map(lambda x: x.race.distance / time_to_number_of_seconds(x.finish_time), ps))
        rankings = list(map(lambda x: get_ranking_from_participation(x) / len(
            self.remove_unranked_participants(x.race.participations)), ps))
        top_3_count = len(list(filter(lambda k: k <= 3, rankings)))

        return [
            np.mean(speeds),
            np.std(speeds),
            top_3_count / len(ps) if len(ps) > 0 else 0,
            np.mean(rankings)
        ]
