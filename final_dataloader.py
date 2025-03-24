# 64 PERFECT FEATURES - NO TURNING BACK. LAST FILE ON DATA LOADING.
import numpy as np

from database import Race, Participation, Horse, Trainer, init_engine, get_session
from datetime import time

from utils import utils

from scipy.stats import norm, lognorm

from functools import lru_cache
from tqdm import tqdm
from datetime import datetime
import os
import json



class FinalDataLoader:
    def __init__(self, size=64, race_number_base=10):
        self.session = get_session()
        self.speeds_mean_std = {}
        self.log_win_odds_mean_std = {}

        self.size = size

        self.race_difficulty_mean = None
        self.race_difficulty_std  = None

        self.race_number_base = race_number_base

        self.scale_data = True
        self.weigh_data = True

    def get_query(self):
        return self.session.query(Race)

    @lru_cache(maxsize=None)
    def get_race_number_difficulty(self, n):
        return np.emath.logn(self.race_number_base, n)

    def load_starting_speeds(self):
        # getting all distances
        distances = set([r.distance for r in self.get_query().all()])

        for distance in distances:
            mean, std = self.get_mean_std_all_speeds(distance)
            self.speeds_mean_std[distance] = (mean, std)


    def get_mean_std_all_speeds(self, distance):
        speeds = []
        races = self.get_query().filter(Race.distance == distance).all()

        for race in tqdm(races, desc=f"Calculating speed for distance = {distance}"):
            ps = get_ps(race)

            for p in ps:
                speeds.append(get_speed(p))

        # remove anomalies
        speeds = np.array(speeds, dtype=np.float64)
        speeds = remove_normal_anomalies(np.array(speeds, dtype=np.float64), threshold=3)

        return np.mean(speeds), np.std(speeds)


    def load_starting_win_odds(self):
        races = self.get_query().all()

        win_odds = {}

        for n in range(4, 15):
            win_odds[n] = []

        for race in tqdm(races, desc="Calculating win-odds"):
            n = get_total_participants(race)
            for p in get_ps(race):
                win_odds[n].append(float(p.win_odds))


        for n in range(4, 15):
            cleaned_win_odds = remove_lognormal_anomalies(np.array(win_odds[n], dtype=np.float64), threshold=3)
            self.log_win_odds_mean_std[n] = (np.mean(np.log(cleaned_win_odds)), np.log(np.std(cleaned_win_odds)))

    def load_mean_std_rating_difficulties(self):
        races = self.get_query().all()
        all_difficulties = []
        for race in tqdm(races, desc="Calculating rating difficulties"):
            ps = get_ps(race)
            target_ratings = [float(p.rating) for p in ps if p.rating is not None]
            if len(target_ratings) < 4:
                continue
            all_difficulties.append(np.mean(target_ratings))

        self.race_difficulty_mean = np.mean(all_difficulties)
        self.race_difficulty_std = np.std(all_difficulties)

    @lru_cache(maxsize=None)
    def get_race_rating_difficulty(self, race):
        mean_rating = get_race_mean_rating(race)
        return norm.cdf(mean_rating, loc=self.race_difficulty_mean, scale=self.race_difficulty_std)

    @lru_cache(maxsize=None)
    def get_speed_score(self, p, distance):
        raw_speed = get_speed(p)
        mean, std = self.speeds_mean_std[distance]
        return norm.cdf(raw_speed, loc=mean, scale=std)

    @lru_cache(maxsize=None)
    def get_win_odds_score(self, p, number):
        raw_win_odds = float(p.win_odds)
        log_mean, log_std = self.log_win_odds_mean_std[number]
        return 1 - lognorm.cdf(raw_win_odds, log_std, scale=np.exp(log_mean))

    def extract_races(self, start_date, end_date, distance):
        races = self.get_query().filter(Race.date >= start_date).filter(Race.date < end_date)
        if distance is not None:
            races = races.filter(Race.distance == distance)
        races = races.all()
        return races

    def scale_jockey_by_horse_rating(self, rating, variable):
        raise NotImplementedError

    def setup(self):
        print("Loading win odds mean and std...")
        self.load_starting_win_odds()
        print("Loading speed mean and std...")
        self.load_starting_speeds()
        print("Loading rating difficulties mean and std")
        self.load_mean_std_rating_difficulties()

    def load_data(
        self,
        start_date,
        end_date,
        distance,
    ):
        races = self.extract_races(start_date, end_date, distance)
        size = self.size

        all_x = {}
        all_y = {}
        horse_nums = {}
        winners = {}
        places = {}

        for race in tqdm(races, desc="Loading data"):
            # remove inexperienced horses
            experienced_ps = get_experienced_ps(race)
            if experienced_ps is None:
                # if not enough experienced horses, remove
                continue
            m = len(experienced_ps)
            participants = get_total_participants(race)

            this_winners = []
            this_places = []
            this_x = np.zeros((m, size), dtype=np.float64)
            this_y = np.zeros((m, 5), dtype=np.float64)
            this_horse_nums = np.zeros(m, dtype=np.int8)

            for i, p in enumerate(experienced_ps):
                p_ranking = get_ranking_from_participation(p)
                this_horse_nums[i] = p.number
                this_y[i, 0] = p_ranking
                this_y[i, 1] = get_seconds_from_time(p.finish_time)
                this_y[i, 2] = self.get_speed_score(p, p.race.distance)
                this_y[i, 3] = p.win_odds
                this_y[i, 4] = participants

                this_x[i] = self.load_p(p, p.horse.trainer_id)

                if p_ranking == 1:
                    this_winners.append(p.number)
                if p_ranking <= 3:
                    this_places.append((p.number, p_ranking))

            this_places.sort(key=lambda x: x[1])
            this_places = [x[0] for x in this_places]

            all_x[race.id] = this_x
            all_y[race.id] = this_y
            horse_nums[race.id] = this_horse_nums
            winners[race.id] = np.array(this_winners, dtype=np.int8)
            places[race.id] = np.array(this_places, dtype=np.int8)

        return all_x, all_y, horse_nums, winners, places

    def get_old_horse_ps(self, horse_id, before):
        session = self.session
        ps = (session.query(Participation)
              .join(Race)
              .filter(Participation.horse_id == horse_id).filter(Race.date < before)
              .order_by(Race.date)
              .all())
        return utils.remove_unranked_participants(ps)

    def get_old_jockey_ps(self, jockey_id, before):
        session = self.session
        ps = (session.query(Participation)
              .join(Race)
              .filter(Participation.jockey_id == jockey_id).filter(Race.date < before)
              .order_by(Race.date)
              .all())
        return utils.remove_unranked_participants(ps)

    def get_old_trainer_ps(self, trainer_id, before):
        session = self.session
        ps = (session.query(Participation)
              .join(Participation.race)
              .join(Participation.horse)
              .join(Horse.trainer)
              .filter(Trainer.id == trainer_id)
              .filter(Race.date < before)
              .order_by(Race.date)
              .all())
        result = utils.remove_unranked_participants(ps)
        return result

    def get_old_jt_combo_ps(self, jockey_id, trainer_id, before):
        session = self.session
        ps = (session.query(Participation)
              .join(Participation.race)
              .join(Participation.horse)
              .join(Horse.trainer)
              .filter(Participation.jockey_id == jockey_id)
              .filter(Trainer.id == trainer_id)
              .filter(Race.date < before)
              .order_by(Race.date)
              .all())
        result = utils.remove_unranked_participants(ps)
        return result

    def scale_by_difficulty(self, values, *weights):
        if self.scale_data:
            acc = values
            for w in weights:
                acc = w * acc
            return acc
        else:
            return values

    def weigh_by_relevancy_mean(self, time_relevancy, track_relevancy, variable, time_weight=0.7, track_weight=0.3):
        if self.weigh_data:
            time_weight, track_weight = normalize_weights(time_weight, track_weight)
            weight = time_weight * time_relevancy + track_weight * track_relevancy
    
            return np.dot(weight, variable) / np.sum(weight)
        else:
            return np.mean(variable)

    def weigh_by_relevancy_std(self, time_relevancy, track_relevancy, variable, weighted_mean, time_weight=0.7, track_weight=0.3):
        if self.weigh_data:
            time_weight, track_weight = normalize_weights(time_weight, track_weight)
            weight = time_weight * time_relevancy + track_weight * track_relevancy

            return np.sqrt(np.dot(weight, np.square(variable - weighted_mean)) / np.sum(weight))
        else:
            return np.std(variable)

    def load_p(self, p, trainer_id):
        trainer = self.session.query(Trainer).filter(Trainer.id == trainer_id).one()

        old_horse_ps = self.get_old_horse_ps(p.horse_id, p.race.date)
        old_jockey_ps = self.get_old_jockey_ps(p.jockey_id, p.race.date)[-50:]
        old_trainer_ps = self.get_old_trainer_ps(trainer_id, p.race.date)[-300:]
        old_jt_combo_ps = self.get_old_jt_combo_ps(p.jockey_id, trainer_id, p.race.date)

        previous_horse_p = old_horse_ps[-1]
        previous_jockey_p = old_jockey_ps[-1]

        horse_ps_info = self.extract_info_from_ps(p.race.date, old_horse_ps)
        jockey_ps_info = self.extract_info_from_ps(p.race.date, old_jockey_ps)
        trainer_ps_info = self.extract_info_from_ps(p.race.date, old_trainer_ps)
        jt_combo_ps_info = self.extract_info_from_ps(p.race.date, old_jt_combo_ps)

        horse_race_rating_difficulty = horse_ps_info[:, 4]
        jockey_race_rating_difficulty = jockey_ps_info[:, 4]
        trainer_race_rating_difficulty = trainer_ps_info[:, 4]
        jt_combo_race_rating_difficulty = jt_combo_ps_info[:, 4]

        horse_race_number_difficulty = horse_ps_info[:, 11]
        jockey_race_number_difficulty = jockey_ps_info[:, 11]
        trainer_race_number_difficulty = trainer_ps_info[:, 11]
        jt_combo_number_difficulty = jt_combo_ps_info[:, 11]

        horse_time_relevancy = horse_ps_info[:, 0]
        jockey_time_relevancy = jockey_ps_info[:, 0]
        trainer_time_relevancy = trainer_ps_info[:, 0]
        jt_combo_time_relevancy = jt_combo_ps_info[:, 0]

        horse_track_relevancy = get_track_relevancy(p.race.distance, p.race.location, get_track_width(p.race), horse_ps_info[:, 1], horse_ps_info[:, 2], horse_ps_info[:, 3])
        jockey_track_relevancy = get_track_relevancy(p.race.distance, p.race.location, get_track_width(p.race), jockey_ps_info[:, 1], jockey_ps_info[:, 2], jockey_ps_info[:, 3])
        trainer_track_relevancy = get_track_relevancy(p.race.distance, p.race.location, get_track_width(p.race), trainer_ps_info[:, 1], trainer_ps_info[:, 2], trainer_ps_info[:, 3])
        jt_combo_track_relevancy = get_track_relevancy(p.race.distance, p.race.location, get_track_width(p.race), jt_combo_ps_info[:, 1], jt_combo_ps_info[:, 2], jt_combo_ps_info[:, 3])

        # Speeds
        horse_max_speed = np.max(horse_ps_info[:, 6])
        horse_mean_weighted_speed = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_ps_info[:, 6])
        horse_std_weighted_speed = self.weigh_by_relevancy_std(horse_time_relevancy, horse_track_relevancy, horse_ps_info[:, 6], horse_mean_weighted_speed)

        jockey_max_speed = np.max(jockey_ps_info[:, 6])
        jockey_mean_weighted_speed = self.weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_ps_info[:, 6])
        jockey_std_weighted_speed = self.weigh_by_relevancy_std(jockey_time_relevancy, jockey_track_relevancy, jockey_ps_info[:, 6], jockey_mean_weighted_speed)

        trainer_mean_weighted_speed = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[:, 6])
        trainer_std_weighted_speed = self.weigh_by_relevancy_std(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[:, 6], trainer_mean_weighted_speed)

        if len(jt_combo_ps_info) > 0:
            jt_combo_mean_weighted_speed = self.weigh_by_relevancy_mean(jt_combo_time_relevancy, jt_combo_track_relevancy, jt_combo_ps_info[:, 6])
            jt_combo_std_weighted_speed = self.weigh_by_relevancy_std(jt_combo_time_relevancy, jt_combo_track_relevancy, jt_combo_ps_info[:, 6], jt_combo_mean_weighted_speed)
        else:
            jt_combo_mean_weighted_speed = 0
            jt_combo_std_weighted_speed = 0

        # Rankings
        horse_score = get_score_from_ranking(get_adjusted_ranking(horse_ps_info[:, 7], horse_ps_info[:, 5]))
        horse_scaled_win_rate = self.scale_by_difficulty((horse_ps_info[:, 7] == 1).astype(np.float64), horse_race_number_difficulty, horse_race_rating_difficulty)
        horse_scaled_place_rate = self.scale_by_difficulty(is_place(horse_ps_info[:, 7], horse_ps_info[:, 5]).astype(np.float64), horse_race_number_difficulty, horse_race_rating_difficulty)
        horse_scaled_score = self.scale_by_difficulty(horse_score, horse_race_rating_difficulty)
        horse_mean_weighted_win_rate = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_win_rate)
        horse_mean_weighted_place_rate = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_place_rate)
        horse_mean_weighted_score = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_score)
        horse_std_weighted_score = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_score, horse_mean_weighted_score)

        jockey_score = get_score_from_ranking(get_adjusted_ranking(jockey_ps_info[:, 7], jockey_ps_info[:, 5]))
        jockey_scaled_win_rate = self.scale_by_difficulty((jockey_ps_info[:, 7] == 1).astype(np.float64), jockey_race_number_difficulty, jockey_race_rating_difficulty)
        jockey_scaled_place_rate = self.scale_by_difficulty(is_place(jockey_ps_info[:, 7], jockey_ps_info[:, 5]).astype(np.float64), jockey_race_number_difficulty, jockey_race_rating_difficulty)
        jockey_scaled_score = self.scale_by_difficulty(jockey_score, jockey_race_rating_difficulty)
        jockey_mean_weighted_win_rate = self.weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_win_rate)
        jockey_mean_weighted_place_rate = self.weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_place_rate)
        jockey_mean_weighted_score = self.weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_score)
        jockey_std_weighted_score = self.weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_score, jockey_mean_weighted_score)

        trainer_score = get_score_from_ranking(get_adjusted_ranking(trainer_ps_info[:, 7], trainer_ps_info[:, 5]))
        trainer_scaled_win_rate = self.scale_by_difficulty((trainer_ps_info[:, 7] == 1).astype(np.float64), trainer_race_number_difficulty, trainer_race_rating_difficulty)
        trainer_scaled_place_rate = self.scale_by_difficulty(is_place(trainer_ps_info[:, 7], trainer_ps_info[:, 5]).astype(np.float64), trainer_race_number_difficulty, trainer_race_rating_difficulty)
        trainer_scaled_score = self.scale_by_difficulty(trainer_score, trainer_race_rating_difficulty)
        trainer_mean_weighted_win_rate = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_scaled_win_rate)
        trainer_mean_weighted_place_rate = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_scaled_place_rate)
        trainer_mean_weighted_score = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_scaled_score)
        trainer_std_weighted_score = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_scaled_score, trainer_mean_weighted_score)

        if len(jt_combo_ps_info) > 0:
            jt_combo_score = get_score_from_ranking(get_adjusted_ranking(jt_combo_ps_info[:, 7], jt_combo_ps_info[:, 5]))
            jt_combo_scaled_win_rate = self.scale_by_difficulty((jt_combo_ps_info[:, 7] == 1).astype(np.float64), jt_combo_number_difficulty, jt_combo_race_rating_difficulty)
            jt_combo_scaled_place_rate = self.scale_by_difficulty(is_place(jt_combo_ps_info[:, 7], jt_combo_ps_info[:, 5]).astype(np.float64), jt_combo_number_difficulty, jt_combo_race_rating_difficulty)
            jt_combo_scaled_score = self.scale_by_difficulty(jt_combo_score, jt_combo_race_rating_difficulty)
            jt_combo_mean_weighted_win_rate = self.weigh_by_relevancy_mean(jt_combo_time_relevancy, jt_combo_track_relevancy, jt_combo_scaled_win_rate)
            jt_combo_mean_weighted_place_rate = self.weigh_by_relevancy_mean(jt_combo_time_relevancy, jt_combo_track_relevancy, jt_combo_scaled_place_rate)
            jt_combo_mean_weighted_score = self.weigh_by_relevancy_mean(jt_combo_time_relevancy, jt_combo_track_relevancy, jt_combo_scaled_score)
            jt_combo_std_weighted_score = self.weigh_by_relevancy_mean(jt_combo_time_relevancy, jt_combo_track_relevancy, jt_combo_scaled_score, jt_combo_mean_weighted_score)
        else:
            jt_combo_mean_weighted_win_rate = 0
            jt_combo_mean_weighted_place_rate = 0
            jt_combo_mean_weighted_score = 0
            jt_combo_std_weighted_score = 0

        # Ratings
        # a) horse should use both rating and rating diff
        # b) trainer should use just rating

        horse_mean_weighted_rating = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_ps_info[:, 8], track_weight=0)
        horse_std_weighted_rating = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_ps_info[:, 8], horse_mean_weighted_rating, track_weight=0)

        horse_delta_rating = get_delta_rating(horse_ps_info[:, 8], p.rating)
        horse_mean_weighted_delta_rating = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_delta_rating)
        horse_std_weighted_delta_rating = self.weigh_by_relevancy_std(horse_time_relevancy, horse_track_relevancy, horse_delta_rating, horse_mean_weighted_delta_rating)

        trainer_mean_weighted_rating = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[:, 8], track_weight=0)
        trainer_std_weighted_rating = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[:, 8], trainer_mean_weighted_rating, track_weight=0)

        # Beaten time
        horse_adjusted_beaten_time = self.scale_by_difficulty(horse_ps_info[:, 10], horse_race_number_difficulty, horse_race_rating_difficulty)
        horse_mean_weighted_beaten_time = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_adjusted_beaten_time)
        horse_std_weighted_beaten_time = self.weigh_by_relevancy_std(horse_time_relevancy, horse_track_relevancy, horse_adjusted_beaten_time, horse_mean_weighted_beaten_time)

        jockey_adjusted_beaten_time = self.scale_by_difficulty(jockey_ps_info[:, 10], jockey_race_number_difficulty, jockey_race_rating_difficulty)
        jockey_mean_weighted_beaten_time = self.weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_adjusted_beaten_time)
        jockey_std_weighted_beaten_time = self.weigh_by_relevancy_std(jockey_time_relevancy, jockey_track_relevancy, jockey_adjusted_beaten_time, jockey_mean_weighted_beaten_time)

        trainer_adjusted_beaten_time = self.scale_by_difficulty(trainer_ps_info[:, 10], trainer_race_number_difficulty, trainer_race_rating_difficulty)
        trainer_mean_weighted_beaten_time = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_adjusted_beaten_time)
        trainer_std_weighted_beaten_time = self.weigh_by_relevancy_std(trainer_time_relevancy, trainer_track_relevancy, trainer_adjusted_beaten_time, trainer_mean_weighted_beaten_time)

        # Win Odds
        horse_adjusted_win_odds = self.scale_by_difficulty(horse_ps_info[:, 9], horse_race_rating_difficulty)
        horse_mean_weighted_win_odds = self.weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_adjusted_win_odds)
        horse_std_weighted_win_odds = self.weigh_by_relevancy_std(horse_time_relevancy, horse_track_relevancy, horse_adjusted_win_odds, horse_mean_weighted_win_odds)

        jockey_adjusted_win_odds = self.scale_by_difficulty(jockey_ps_info[:, 9], jockey_race_rating_difficulty)
        jockey_mean_weighted_win_odds = self.weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_adjusted_win_odds)
        jockey_std_weighted_win_odds = self.weigh_by_relevancy_std(jockey_time_relevancy, jockey_track_relevancy, jockey_adjusted_win_odds, jockey_mean_weighted_win_odds)

        trainer_adjusted_win_odds = self.scale_by_difficulty(trainer_ps_info[:, 9], trainer_race_rating_difficulty)
        trainer_mean_weighted_win_odds = self.weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_adjusted_win_odds)
        trainer_std_weighted_win_odds = self.weigh_by_relevancy_std(trainer_time_relevancy, trainer_track_relevancy, trainer_adjusted_win_odds, trainer_mean_weighted_win_odds)

        return np.array([
            # independent features (6)
            p.rating,
            p.number,
            p.lane,
            p.horse_weight,
            p.gear_weight,
            p.gear_weight / (p.horse_weight + p.horse_weight),

            # previous horse stats (6)
            self.get_speed_score(previous_horse_p, previous_horse_p.race.distance),
            p.rating - (previous_horse_p.rating if previous_horse_p.rating is not None else get_race_mean_rating(previous_horse_p.race)),
            self.get_win_odds_score(previous_horse_p, get_total_participants(previous_horse_p.race)),
            (p.race.date - previous_horse_p.race.date).days,
            self.get_adjusted_beaten_time(previous_horse_p),
            p.horse_weight - previous_horse_p.horse_weight,

            # previous jockey stats (3)
            self.get_speed_score(previous_jockey_p, previous_jockey_p.race.distance),
            (p.race.date - previous_jockey_p.race.date).days,
            self.get_adjusted_beaten_time(previous_jockey_p),

            # horse historic data (16)
            len(old_horse_ps),
            horse_max_speed,
            horse_mean_weighted_speed,
            horse_std_weighted_speed,
            horse_mean_weighted_win_rate,
            horse_mean_weighted_place_rate,
            horse_mean_weighted_score,
            horse_std_weighted_score,
            horse_mean_weighted_rating,
            horse_std_weighted_rating,
            horse_mean_weighted_delta_rating,
            horse_std_weighted_delta_rating,
            horse_mean_weighted_beaten_time,
            horse_std_weighted_beaten_time,
            horse_mean_weighted_win_odds,
            horse_std_weighted_win_odds,

            # jockey historic data (13)
            len(old_jockey_ps),
            jockey_max_speed,  # max scaled speed (by horse rating)
            np.max(jockey_ps_info[:, 6]),  # max adjusted speed
            jockey_mean_weighted_speed,
            jockey_std_weighted_speed,
            jockey_mean_weighted_win_rate,
            jockey_mean_weighted_place_rate,
            jockey_mean_weighted_score,
            jockey_std_weighted_score,
            jockey_mean_weighted_beaten_time,
            jockey_std_weighted_beaten_time,
            jockey_mean_weighted_win_odds,
            jockey_std_weighted_win_odds,

            # trainer historic data (13)
            len(trainer.horses),
            trainer_mean_weighted_speed,
            trainer_std_weighted_speed,
            trainer_mean_weighted_win_rate,
            trainer_mean_weighted_place_rate,
            trainer_mean_weighted_score,
            trainer_std_weighted_score,
            trainer_mean_weighted_rating,
            trainer_std_weighted_rating,
            trainer_mean_weighted_beaten_time,
            trainer_std_weighted_beaten_time,
            trainer_mean_weighted_win_odds,
            trainer_std_weighted_win_odds,

            # jockey x trainer combo (7)
            len(old_jt_combo_ps),
            jt_combo_mean_weighted_speed,
            jt_combo_std_weighted_speed,
            jt_combo_mean_weighted_win_rate,
            jt_combo_mean_weighted_place_rate,
            jt_combo_mean_weighted_score,
            jt_combo_std_weighted_score,
        ], dtype=np.float64)

    def get_adjusted_beaten_time(self, p):
        standard_distance = 1200
        distance = p.race.distance
        winner_p = get_winner(p.race)

        if distance == standard_distance:
            return get_seconds_from_time(p.finish_time) - get_seconds_from_time(winner_p.finish_time)
        else:
            this_p_speed_score = self.get_speed_score(p, distance)
            winner_speed_score = self.get_speed_score(winner_p, distance)

            # use 1200 as standard
            speed_mean_standard, speed_std_standard = self.speeds_mean_std[standard_distance]
            this_p_adjusted_time = standard_distance / norm.ppf(this_p_speed_score, speed_mean_standard, speed_std_standard)
            winner_adjusted_time = standard_distance / norm.ppf(winner_speed_score, speed_mean_standard, speed_std_standard)

            return this_p_adjusted_time - winner_adjusted_time


    def extract_info_from_ps(self, race_date, ps):
        """Gets adjusted data"""
        n = len(ps)
        result = np.zeros((n, 12), dtype=np.float64)

        for i, p in enumerate(ps):
            distance = p.race.distance
            participants = get_total_participants(p.race)
            result[i, 0] = get_time_relevancy(race_date, p.race.date, time_decay=0.03)
            result[i, 1] = distance
            result[i, 2] = encode_location(p.race.location)
            result[i, 3] = get_track_width(p.race)
            result[i, 4] = self.get_race_rating_difficulty(p.race)
            result[i, 5] = participants
            result[i, 6] = self.get_speed_score(p, distance)
            result[i, 7] = get_ranking_from_participation(p)
            result[i, 8] = p.rating if p.rating is not None else get_race_mean_rating(p.race)
            result[i, 9] = self.get_win_odds_score(p, participants)
            result[i, 10] = self.get_adjusted_beaten_time(p)
            result[i, 11] = self.get_race_number_difficulty(participants)

        return result


@lru_cache(maxsize=None)
def get_ps(race):
    return utils.remove_unranked_participants(race.participations)


def get_ranking_from_participation(x):
    return int(x.ranking.replace("DH", "").strip())


def get_winner(race: Race) -> Participation:
    # if multiple, return first one (rare case)
    ps = get_ps(race)
    result = min(ps, key=lambda p: get_ranking_from_participation(p))
    return result


def get_seconds_from_time(t: time) -> float:
    return t.minute * 60 + t.second + t.microsecond * 1e-6


@lru_cache(maxsize=None)
def get_speed(p: Participation) -> float:
    # time in m/s
    distance = p.race.distance
    seconds = get_seconds_from_time(p.finish_time)
    return distance / seconds


def get_adjusted_ranking(ranking, total):
    return ranking / (total - 1)


def get_score_from_ranking(ranking, decay_sharpness=1.5):
    base_log = np.log(1 + decay_sharpness)
    a = np.log(decay_sharpness) * base_log / (base_log - np.log(decay_sharpness))
    return a / np.log(ranking + decay_sharpness) - a / base_log


@lru_cache(maxsize=None)
def get_race_mean_rating(race: Race):
    ps = get_ps(race)
    ratings = [p.rating for p in ps if p.rating is not None]
    if len(ratings) == 0:
        return 25
    else:
        return np.mean(ratings)


@lru_cache(maxsize=None)
def encode_location(location):
    match location:
        case "Sha Tin": return 0
        case "Happy Valley": return 1
        case _: raise Exception(f"Unknown location {location}")


@lru_cache(maxsize=None)
def get_time_relevancy(current_date, old_date, time_decay=0.01) -> float:
    days_diff = (current_date - old_date).days
    return np.exp(-time_decay * days_diff)


@lru_cache(maxsize=None)
def get_total_participants(race):
    ps = race.participations
    return len(utils.remove_unranked_participants(ps))


def get_experienced_ps(race):
    race_ps = utils.remove_unranked_participants(race.participations)
    result = [p for p in race_ps if not utils.is_new_horse(p) and not utils.is_new_jockey(p) and p.rating is not None]
    result.sort(key=lambda x: x.number)
    valid_count = len(result)
    if valid_count >= 4:
        return result
    else:
        return None


@lru_cache(maxsize=None)
def get_track_width(race: Race):
    location = race.location
    course = race.course

    sha_tin_track_width = {
        "A": 30.5,
        "A+2": 28.5,
        "A+3": 27.5,
        "B": 26,
        "B+2": 24,
        "C": 21.3,
        "C+3": 18.3,
        "ALL WEATHER TRACK": 22.8
    }

    hv_track_width = {
        "A": 30.5,
        "A+2": 28.5,
        "B": 26.5,
        "B+2": 24.5,
        "C": 22.5,
        "C+3": 19.5,
    }

    if course.upper() == "ALL WEATHER TRACK":
        return sha_tin_track_width[course.upper()]

    letter = course.split('"')[1]  # get middle element
    if location == "Sha Tin":
        return sha_tin_track_width[letter]
    elif location == "Happy Valley":
        return hv_track_width[letter]
    else:
        raise Exception(f"Unknown location: {location}")


def is_place(ranking, participants):
    result = ranking <= 2 | ((ranking == 3) & (participants >= 7))
    return result


def normalize_weights(*weights):
    total = sum(weights)
    result = []
    for w in weights:
        result.append(w / total)
    return result


def get_track_relevancy(
    race_distance,
    race_location,
    race_track_width,
    distance,
    location,
    track_width,
    distance_decay=0.05,
    distance_weight=0.7,
    location_weight=0.05,
    track_width_weight=0.25,
):
    distance_weight, location_weight, track_width_weight = normalize_weights(distance_weight, location_weight, track_width_weight)

    distance_relevancy = np.exp(-distance_decay * np.abs(distance - race_distance))
    location_relevancy = (location == race_location).astype(np.float64)
    width_relevancy = 1 - np.abs(track_width - race_track_width) / race_track_width

    total_weight = distance_relevancy * distance_weight + location_relevancy * location_weight + width_relevancy * track_width_weight
    return total_weight


def get_delta_rating(rating_hist, curr_rating):
    n = rating_hist.shape[0]
    result = np.zeros(n, dtype=np.float64)
    result[:n-1] = np.diff(rating_hist)
    result[n - 1] = curr_rating - rating_hist[-1]
    return result


def remove_normal_anomalies(values, threshold=3):
    mean = np.mean(values)
    std = np.std(values)
    scaled = (values - mean) / std

    return values[np.where(np.abs(scaled) <= threshold)]


def remove_lognormal_anomalies(values, threshold=3):
    log_values = np.log(values)
    mean = np.mean(log_values)
    std = np.std(log_values)

    scaled = (log_values - mean) / std
    return values[np.where(np.abs(scaled) <= threshold)]


""" Ideas:
Note: everything here should be adjusted by time relevancy and race relevancy
1. Capturing rating changes -> rating penalty / increase indicates how well that race went for that horses, (maybe) adjust it by race difficulty
2. Capturing previous speeds -> adjust it by speed factor
3. Capturing previous beaten lengths -> (maybe) adjust it by speed factor
4. Capturing ranking -> adjust it by race difficulty
5. Capturing win odds change ->

Factors that can affect speed
1. Distance
2. Condition
3. Lane width
4. Location
5. Lane number
"""


def save_data(directory, all_x, all_y, horse_nums, wins, places):
    os.makedirs(directory, exist_ok=True)

    np.savez(f"{directory}/data_x.npz", **all_x)
    np.savez(f"{directory}/data_y.npz", **all_y)
    np.savez(f"{directory}/horse_nums.npz", **horse_nums)
    np.savez(f"{directory}/wins.npz", **wins)
    np.savez(f"{directory}/places.npz", **places)


def load_all_for_distance(dataloader: FinalDataLoader, distance, directory):
    # both
    dataloader.scale_data = True
    dataloader.weigh_data = True
    both_path = os.path.join(directory, "both")
    load_train_test_data(dataloader, both_path, distance)

    # scaled
    dataloader.weigh_data = False
    scaled_path = os.path.join(directory, "scaled")
    load_train_test_data(dataloader, scaled_path, distance)

    # weighed
    dataloader.weigh_data = True
    dataloader.scale_data = False
    weighed_path = os.path.join(directory, "weighed")
    load_train_test_data(dataloader, weighed_path, distance)

    # neither
    dataloader.weigh_data = False
    neither_path = os.path.join(directory, "neither")
    load_train_test_data(dataloader, neither_path, distance)


def load_train_test_data(dataloader, directory, distance):
    train_start_date = datetime(2012, 9, 1).date()
    train_end_date = datetime(2022, 9, 1).date()
    test_start_date = datetime(2022, 9, 1).date()
    test_end_date = datetime(2025, 9, 1).date()

    train_x, train_y, horse_nums, wins, places = dataloader.load_data(train_start_date, train_end_date, distance)
    test_x, test_y, test_horse_nums, test_wins, test_places = dataloader.load_data(test_start_date, test_end_date, distance)

    train_dir = os.path.join(directory, "train")
    test_dir = os.path.join(directory, "test")

    save_data(train_dir, train_x, train_y, horse_nums, wins, places)
    save_data(test_dir, test_x, test_y, test_horse_nums, test_wins, test_places)


def main():
    init_engine()
    dataloader = FinalDataLoader()

    dataloader.setup()

    distances = [None, 1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400]
    distances_path = ["combined"] + [f"distance_{d}" for d in distances[1:]]

    for i in range(len(distances)):
        this_distance = distances[i]
        this_path = distances_path[i]

        load_all_for_distance(dataloader, this_distance, this_path)


if __name__ == "__main__":
    main()

