import os
from datetime import timedelta

import numpy as np

from database import Participation, Horse, Jockey, Trainer
from database.fetch import Fetch
from dataloader.loader import Loader
from dataloader.utils import *

from tqdm import tqdm

import utils.utils as utils


class SimpleLoader(Loader):
    def __init__(self):
        super().__init__()
        self.input_features = 20
        self.zscore_indices = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18]

    @staticmethod
    def get_participation_data(p: Participation):
        return [
            p.lane,
            p.gear_weight,
            p.rating if p.rating is not None else 25,
            p.horse_weight,
            p.win_odds
        ]

    @staticmethod
    def get_horse_data(h: Horse):
        return [
            h.age  # TODO: subtract now from race date
        ]

    @staticmethod
    def get_race_data(r: Race):
        return [
            convert_race_class(r.race_class),
            r.distance,
            r.total_bet
        ]

    @staticmethod
    def get_grouped_stats(ps: list[Participation]):
        speeds = list(map(lambda x: x.race.distance / time_to_number_of_seconds(x.finish_time), ps))
        rankings = list(map(lambda x: get_ranking_from_participation(x) / len(
            utils.remove_unranked_participants(x.race.participations)), ps))
        top_3_count = len(list(filter(lambda k: k <= 3, rankings)))

        return [
            np.mean(speeds),
            np.std(speeds),
            top_3_count / len(ps) if len(ps) > 0 else 0,
            np.mean(rankings)
        ]

    def _load_from_db(self, session) -> (np.ndarray, np.ndarray):
        ps = utils.remove_unranked_participants(session.query(Participation).all())
        m = len(ps)
        result_x = np.zeros((m, self.input_features), dtype=np.float32)
        result_y = np.array(list(map(get_ranking_from_participation, ps[:m])), dtype=np.float32)

        for idx in tqdm(range(m), desc="Loading data"):
            p = ps[idx]
            res = self.load_one_participation(session, p)
            result_x[idx, :] = np.copy(res)

        return result_x, result_y

    def train_normalize(self, train_x):
        train_mean = np.mean(train_x, axis=0)
        train_std = np.std(train_x, axis=0)
        train_std[train_std == 0] = 1
        self.normalize(train_x, train_mean=train_mean, train_std=train_std)

        return {
            "train_mean": train_mean,
            "train_std": train_std
        }

    def normalize(self, x, **kwargs):
        try:
            train_mean = kwargs['train_mean']
            train_std = kwargs['train_std']
        except KeyError as e:
            print("Missing train_mean or train_std argument")
            raise e

        for i in range(x.shape[1]):
            if i in self.zscore_indices:
                x[:, i] = (x[:, i] - train_mean[i]) / train_std[i]

    @staticmethod
    def get_relevant_participations(session, before, after_days_count, horse_id, jockey_id, trainer_id):
        after = before - timedelta(after_days_count)

        horse_ps = get_relevant_participation(session, before, after, horse_id=horse_id)
        jockey_ps = get_relevant_participation(session, before, after, jockey_id=jockey_id)
        trainer_ps = get_relevant_participation(session, before, after, trainer_id=trainer_id)

        return horse_ps, jockey_ps, trainer_ps

    def load_one_participation(self, session, p: Participation):
        static_data = (SimpleLoader.get_participation_data(p)
                       + SimpleLoader.get_race_data(p.race)
                       )

        horse_ps, jockey_ps, trainer_ps = self.get_relevant_participations(session,
            p.race.date, 90, p.horse_id, p.jockey_id, p.horse.trainer_id)

        result = np.array(static_data
                          + SimpleLoader.get_grouped_stats(horse_ps)
                          + SimpleLoader.get_grouped_stats(jockey_ps)
                          + SimpleLoader.get_grouped_stats(trainer_ps),
                          dtype=np.float32)

        result = np.nan_to_num(result)
        return result

    def load_predict(self, fetch, data):
        result = np.zeros((len(data), self.input_features), dtype=np.float32)
        for idx, d in enumerate(data):
            result[idx] = np.copy(self.load_one_predict(fetch, **d))
        return result

    def load_one_predict(self, session, **kwargs):
        """Load one prediction entry"""
        horse_id = kwargs["horse_id"]
        jockey_id = kwargs["jockey_id"]
        date = kwargs["date"]
        trainer_id = kwargs["trainer_id"]
        gear_weight = kwargs["gear_weight"]
        horse_weight = kwargs["horse_weight"]
        lane = kwargs["lane"]
        race_class = kwargs["race_class"]      # convert_race_class should have already been called
        distance = kwargs["distance"]
        total_bet = kwargs["total_bet"]
        win_odds = kwargs["win_odds"]

        entry = np.zeros(self.input_features, dtype=np.float32)
        entry[0] = lane
        entry[1] = gear_weight
        entry[2] = horse_weight
        entry[3] = win_odds
        entry[4] = race_class
        entry[5] = distance
        entry[6] = total_bet

        horse_ps, jockey_ps, trainer_ps = self.get_relevant_participations(session,
            date, 90, horse_id, jockey_id, trainer_id)

        entry[7:11] = np.nan_to_num(np.array(SimpleLoader.get_grouped_stats(horse_ps), dtype=np.float32))
        entry[11:15] = np.nan_to_num(np.array(SimpleLoader.get_grouped_stats(jockey_ps), dtype=np.float32))
        entry[15:19] = np.nan_to_num(np.array(SimpleLoader.get_grouped_stats(trainer_ps), dtype=np.float32))

        return entry
