import os
from datetime import timedelta

import numpy as np

from database import Participation, Horse, Jockey, Trainer
from database.fetch import Fetch
from dataloader.loader import Loader
from dataloader.utils import *

from tqdm import tqdm

import utils.utils as utils


class ParticipationRankingLoader(Loader):
    def __init__(self):
        super().__init__()
        self.input_features = 77
        self.zscore_indices = [0, 1, 6]

    def __repr__(self):
        return "Participation Ranking Loader"

    @staticmethod
    def get_participation_data(p: Participation):
        return [
            p.lane,
            p.number,
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
    def get_mean_std_max_min(data):
        data = list(filter(lambda x: x is not None, data))
        if len(data) == 0:
            return [0, 0]
        else:
            return [
                np.mean(data),
                np.std(data),
                # np.max(data),
                # np.min(data)
            ]

    @staticmethod
    def calculate_change_and_derivative_by_date(values, dates):
        filtered_data = [(v, d) for v, d in zip(values, dates) if v is not None]
        if not filtered_data:  # Handle case where all values are None
            return np.array([]), np.array([])

        # Separate filtered values and dates
        filtered_values, filtered_dates = zip(*filtered_data)
        filtered_values = np.array(filtered_values)
        filtered_dates = np.array(filtered_dates)

        # Calculate differences
        difference = np.diff(filtered_values)
        dates_diff = np.diff(filtered_dates)

        # Convert timedelta to days
        days_f = np.vectorize(lambda x: x.days)
        if len(dates_diff) > 0:
            days_diff = days_f(dates_diff)
        else:
            days_diff = np.array([])

        # Avoid division by zero
        days_diff[days_diff == 0] = 1
        derivative = difference / days_diff

        return difference, derivative

    @staticmethod
    def get_grouped_stats(ps: list[Participation]):
        # sort participations by racing date
        ps.sort(key=lambda x: x.race.date, reverse=True)

        dates = list(map(lambda x: x.race.date, ps))

        speeds = list(map(lambda x: x.race.distance / time_to_number_of_seconds(x.finish_time), ps))
        speeds_diff, speeds_derivative = (
            ParticipationRankingLoader.calculate_change_and_derivative_by_date(speeds, dates))

        rankings = list(map(lambda x: get_ranking_from_participation(x) / len(
            utils.remove_unranked_participants(x.race.participations)), ps))

        top_1_count = len(list(filter(lambda k: k == 1, rankings)))
        top_2_count = len(list(filter(lambda k: k <= 2, rankings)))
        top_3_count = len(list(filter(lambda k: k <= 3, rankings)))
        top_4_count = len(list(filter(lambda k: k <= 4, rankings)))

        # rating
        ratings = list(map(lambda x: x.rating, ps))
        ratings_diff, ratings_derivative = (
            ParticipationRankingLoader.calculate_change_and_derivative_by_date(ratings, dates)
        )

        # weight changes
        horse_weights = list(map(lambda x: x.horse_weight, ps))
        horse_weights_diff, horse_weights_derivative = (
            ParticipationRankingLoader.calculate_change_and_derivative_by_date(horse_weights, dates)
        )

        # win odds
        win_odds = list(map(lambda x: x.win_odds, ps))
        win_odds_diff, win_odds_derivative = (
            ParticipationRankingLoader.calculate_change_and_derivative_by_date(win_odds, dates)
        )

        result = [
            # number of participations
            len(ps),

            # latest speed stuff
            0 if len(speeds) == 0 else speeds[0],
            0 if len(speeds_diff) == 0 else speeds_diff[0],
            0 if len(speeds_derivative) == 0 else speeds_derivative[0],

            # top performance ratio
            top_1_count / len(ps) if len(ps) > 0 else 0,
            top_2_count / len(ps) if len(ps) > 0 else 0,
            top_3_count / len(ps) if len(ps) > 0 else 0,
            top_4_count / len(ps) if len(ps) > 0 else 0,

            # latest ranking
            0 if len(ps) == 0 else get_ranking_from_participation(ps[0]),

            # latest rating
            0 if len(ratings) == 0 else ratings[0],
            # 0 if len(ratings_diff) == 0 else ratings_diff[0],
            # 0 if len(ratings_derivative) == 0 else ratings_derivative[0],

            # latest horse weight
            0 if len(horse_weights) == 0 else horse_weights[0],
            # 0 if len(horse_weights_diff) == 0 else horse_weights_diff[0],
            # 0 if len(horse_weights_derivative) == 0 else horse_weights_derivative[0],

            # latest win odds
            0 if len(win_odds) == 0 else win_odds[0],
            # 0 if len(win_odds_diff) == 0 else win_odds_diff[0],
            # 0 if len(win_odds_derivative) == 0 else win_odds_derivative[0],
        ]

        result += ParticipationRankingLoader.get_mean_std_max_min(rankings)
        result += ParticipationRankingLoader.get_mean_std_max_min(speeds)
        # result += ParticipationRankingLoader.get_mean_std_max_min(speeds_diff)
        # result += ParticipationRankingLoader.get_mean_std_max_min(speeds_derivative)
        result += ParticipationRankingLoader.get_mean_std_max_min(ratings)
        # result += ParticipationRankingLoader.get_mean_std_max_min(ratings_diff)
        # result += ParticipationRankingLoader.get_mean_std_max_min(ratings_derivative)
        result += ParticipationRankingLoader.get_mean_std_max_min(horse_weights)
        # result += ParticipationRankingLoader.get_mean_std_max_min(horse_weights_diff)
        # result += ParticipationRankingLoader.get_mean_std_max_min(horse_weights_derivative)
        result += ParticipationRankingLoader.get_mean_std_max_min(win_odds)
        # result += ParticipationRankingLoader.get_mean_std_max_min(win_odds_diff)
        # result += ParticipationRankingLoader.get_mean_std_max_min(win_odds_derivative)

        return result

    @staticmethod
    def load_opponent_stats(session, p: Participation):
        date = p.race.date
        opponents = p.race.participations
        opponents = utils.remove_unranked_participants(opponents)

        num_opponents = len(opponents) - 1

        all_stats = np.zeros((num_opponents, 210), dtype=np.float32)
        for idx, opponent in enumerate(opponents):
            # ignore if same horse
            if p.horse_id == opponent.horse_id:
                continue

            horse_ps, jockey_ps, trainer_ps = ParticipationRankingLoader.get_relevant_participations(
                session, date, 90, opponent.horse_id, opponent.jockey_id, opponent.horse.trainer_id)

            all_stats[idx, :70]     = ParticipationRankingLoader.get_grouped_stats(horse_ps)
            all_stats[idx, 70:140]  = ParticipationRankingLoader.get_grouped_stats(jockey_ps)
            all_stats[idx, 140:210] = ParticipationRankingLoader.get_grouped_stats(trainer_ps)

        result_mean = np.mean(all_stats, axis=1)

        return result_mean


    def _load_from_db(self, session, start_date=None, end_date=None) -> (np.ndarray, np.ndarray):
        ps = session.query(Participation).join(Race)

        if start_date is not None:
            ps = ps.filter(Race.date >= start_date)

        if end_date is not None:
            ps = ps.filter(Race.date < end_date)

        ps = ps.all()
        ps = utils.remove_unranked_participants(ps)
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
            if i not in self.zscore_indices:
                x[:, i] = (x[:, i] - train_mean[i]) / train_std[i]

    @staticmethod
    def get_relevant_participations(session, before, after_days_count, horse_id, jockey_id, trainer_id):
        after = before - timedelta(after_days_count)

        horse_ps = get_relevant_participation(session, before, after, horse_id=horse_id)
        jockey_ps = get_relevant_participation(session, before, after, jockey_id=jockey_id)
        trainer_ps = get_relevant_participation(session, before, after, trainer_id=trainer_id)

        return horse_ps, jockey_ps, trainer_ps

    @staticmethod
    def load_one_participation(session, p: Participation):
        static_data = (ParticipationRankingLoader.get_participation_data(p)
                       + ParticipationRankingLoader.get_race_data(p.race)
                       )

        horse_ps, jockey_ps, trainer_ps = ParticipationRankingLoader.get_relevant_participations(session,
            p.race.date, 90, p.horse_id, p.jockey_id, p.horse.trainer_id)

        if len(horse_ps) > 0:
            latest_horse_p  = max(horse_ps, key=lambda x: x.race.date).race.date
        else:
            latest_horse_p  = p.race.date

        if len(jockey_ps) > 0:
            latest_jockey_p = max(jockey_ps, key=lambda x: x.race.date).race.date
        else:
            latest_jockey_p = p.race.date

        num_days_horse  = (p.race.date - latest_horse_p).days
        num_days_jockey = (p.race.date - latest_jockey_p).days

        result = np.array(static_data + [num_days_horse, num_days_jockey]
                          + ParticipationRankingLoader.get_grouped_stats(horse_ps)
                          + ParticipationRankingLoader.get_grouped_stats(jockey_ps)
                          + ParticipationRankingLoader.get_grouped_stats(trainer_ps),
                          dtype=np.float32)

        result = np.nan_to_num(result)
        return result

    def load_predict(self, session, data):
        result = np.zeros((len(data), self.input_features), dtype=np.float32)
        for idx, d in enumerate(data):
            result[idx] = np.copy(self.load_one_predict(session, **d))
        return result

    def load_one_predict(self, session, **kwargs):
        """Load one prediction entry"""
        horse_id = kwargs["horse_id"]
        jockey_id = kwargs["jockey_id"]
        number = kwargs["number"]
        date = kwargs["date"]
        rating = kwargs["rating"]
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
        entry[1] = number
        entry[2] = gear_weight
        entry[3] = rating if rating is not None else 25
        entry[4] = horse_weight
        entry[5] = win_odds
        entry[6] = race_class
        entry[7] = distance
        entry[8] = total_bet

        horse_ps, jockey_ps, trainer_ps = self.get_relevant_participations(session,
            date, 90, horse_id, jockey_id, trainer_id)

        if len(horse_ps) > 0:
            latest_horse_p = max(horse_ps, key=lambda x: x.race.date).race.date
        else:
            latest_horse_p = date

        if len(jockey_ps) > 0:
            latest_jockey_p = max(jockey_ps, key=lambda x: x.race.date).race.date
        else:
            latest_jockey_p = date

        num_days_horse = (date - latest_horse_p).days
        num_days_jockey = (date - latest_jockey_p).days

        entry[9] = num_days_horse
        entry[10] = num_days_jockey

        entry[11:55] = np.nan_to_num(np.array(ParticipationRankingLoader.get_grouped_stats(horse_ps), dtype=np.float32))
        entry[55:99] = np.nan_to_num(np.array(ParticipationRankingLoader.get_grouped_stats(jockey_ps), dtype=np.float32))
        entry[99:143] = np.nan_to_num(np.array(ParticipationRankingLoader.get_grouped_stats(trainer_ps), dtype=np.float32))

        return entry
