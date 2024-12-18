import os
from datetime import timedelta

import numpy as np

from database import Participation, Horse, fetch, Jockey, Trainer
from dataloader.loader import Loader
from dataloader.utils import *

from tqdm import tqdm

import utils.utils as utils


class SimpleLoader(Loader):
    def __init__(self, save_dir=None):
        super().__init__(save_dir)
        self.zscore_indices = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18]

    @staticmethod
    def get_participation_data(p: Participation):
        return [
            p.lane,
            p.gear_weight,
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

    def _load_from_dir(self, directory):
        x_path = os.path.join(directory, "data_x.npy")
        y_path = os.path.join(directory, "data_y.npy")
        data_x = np.load(x_path)
        data_y = np.load(y_path)

        return data_x, data_y

    def _load_from_db(self) -> (np.ndarray, np.ndarray):
        ps = utils.remove_unranked_participants(self.session.query(Participation).all())
        m = len(ps)
        result_x = np.zeros((m, 19), dtype=np.float32)
        result_y = np.array(list(map(get_ranking_from_participation, ps[:m])), dtype=np.float32)
        result_y = (result_y <= 3).astype(np.float32)

        for idx, p in tqdm(enumerate(ps[:m]), desc="Loading data"):
            res = self.load_one_participation(p)
            result_x[idx, :] = np.copy(res)

        return result_x, result_y

    def train_normalize(self, train_x):
        train_mean = np.mean(train_x, axis=0)
        train_std = np.std(train_x, axis=0)
        train_std[train_x == 0] = 1
        self.normalize(train_x, train_mean=train_mean, train_std=train_std)

        return train_mean, train_std

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

    def get_relevant_participations(self, before, after_days_count, horse_id, jockey_id, trainer_id):
        after = before - timedelta(after_days_count)

        relevant_ps = (self.session.query(Participation).join(Race).join(Horse)
                       .filter(Race.date < before)
                       .filter(Race.date >= after)
                       )

        horse_ps = relevant_ps.filter(Participation.horse_id == horse_id).all()
        jockey_ps = relevant_ps.filter(Participation.jockey_id == jockey_id).all()
        trainer_ps = relevant_ps.filter(Horse.trainer_id == trainer_id).all()

        return horse_ps, jockey_ps, trainer_ps

    def load_one_participation(self, p: Participation):
        static_data = (SimpleLoader.get_participation_data(p)
                       + SimpleLoader.get_race_data(p.race)
                       )

        horse_ps, jockey_ps, trainer_ps = self.get_relevant_participations(
            p.race.date, 90, p.horse_id, p.jockey_id, p.horse.trainer_id)

        result = np.array(static_data
                          + SimpleLoader.get_grouped_stats(utils.remove_unranked_participants(horse_ps))
                          + SimpleLoader.get_grouped_stats(utils.remove_unranked_participants(jockey_ps))
                          + SimpleLoader.get_grouped_stats(utils.remove_unranked_participants(trainer_ps)),
                          dtype=np.float32)

        result = np.nan_to_num(result)
        return result

    def load_predict(self, **kwargs):
        """Load one prediction entry"""
        horse_id = kwargs["horse_id"]
        jockey_name = kwargs["jockey_name"]
        date = kwargs["date"]
        trainer_name = kwargs["trainer_name"]
        gear_weight = kwargs["gear_weight"]
        horse_weight = kwargs["horse_weight"]
        win_odds = kwargs["win_odds"]
        lane = kwargs["lane"]
        race_class = kwargs["race_class"]      # convert_race_class should have already been called
        distance = kwargs["distance"]
        total_bet = kwargs["total_bet"]

        entry = np.zeros(19, dtype=np.float32)
        entry[0] = lane
        entry[1] = gear_weight
        entry[2] = horse_weight
        entry[3] = win_odds
        entry[4] = race_class
        entry[5] = distance
        entry[6] = total_bet

        js = fetch.FetchJockey.filter(Jockey.name == jockey_name).all()
        ts = fetch.FetchTrainer.filter(Trainer.name == trainer_name).all()

        jockey_id = None if len(js) == 0 else js[0].id
        trainer_id = None if len(ts) == 0 else ts[0].id

        horse_ps, jockey_ps, trainer_ps = self.get_relevant_participations(
            date, 90, horse_id, jockey_id, trainer_id)

        entry[7:11] = np.nan_to_num(np.array(SimpleLoader.get_grouped_stats(horse_ps), dtype=np.float32))
        entry[11:15] = np.nan_to_num(np.array(SimpleLoader.get_grouped_stats(jockey_ps), dtype=np.float32))
        entry[15:19] = np.nan_to_num(np.array(SimpleLoader.get_grouped_stats(trainer_ps), dtype=np.float32))

        return entry
