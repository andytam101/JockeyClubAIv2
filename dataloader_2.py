from feature_selection import *
from dataloader.utils import *
from database import init_engine, get_session, Race, Participation, Trainer
from utils.pools import WIN, PLACE

import numpy as np
import os
from datetime import datetime
from tqdm import tqdm


def calculate_race_difficulty(race, rating_weight=0.7, number_weight=0.3):
    ps = get_relevant_ps_from_race(race)
    opponent_ratings = [this_p.rating for this_p in ps if this_p.rating is not None]
    if len(opponent_ratings) == 0:
        opponent_ratings = [25]
    rating_mean = np.mean(opponent_ratings)
    normalized_number = len(ps) / 14
    return rating_weight * rating_mean + number_weight * normalized_number


def get_experienced_ps(race):
    race_ps = utils.remove_unranked_participants(race.participations)
    result = [p for p in race_ps if not is_new_horse(p) and not is_new_jockey(p) and p.rating is not None]
    result.sort(key=lambda x: x.number)
    total_count = len(race_ps)
    valid_count = len(result)
    if (total_count - valid_count) < 3 and valid_count >= 5:
        return result
    else:
        return None


def calculate_weighted_mean_speed(days, speed, time_decay=0.03):
    weights = np.exp(-time_decay * days)
    # prevent zero division error
    return np.dot(weights, speed) / (np.sum(weights) + 1e-6)


def calculate_weighted_ranking(
    days,
    difficulties,
    rankings,
    mode = None,
    time_decay=0.03,
    time_weight=0.3,
    diff_weight=0.7,
):
    if mode == WIN:
        rankings = (rankings == 1).astype(np.float32)
    elif mode == PLACE:
        rankings = (rankings <= 3).astype(np.float32)

    time_factor = np.exp(-time_decay * days)
    diff_factor = difficulties
    overall_weight = time_weight * time_factor + diff_weight * diff_factor
    return np.mean(overall_weight * rankings)


def calculate_weighted_rating(
    days,
    difficulties,
    ratings,
    time_decay=0.03,
    time_weight=0.3,
    diff_weight=0.7,
):
    rating_diffs = np.diff(ratings)
    time_factor = np.exp(-time_decay * days)
    diff_factor = difficulties
    overall_weight = time_weight * time_factor + diff_weight * diff_factor
    return overall_weight * rating_diffs


class DataLoader:
    def __init__(self, size, distance):
        self.size = size
        self.distance = distance
        self.session = get_session()

        # caches
        self.race_difficulties = {}
        self.race_participants = {}
        self.race_ps = {}
        self.race_ratings = {}
        self.old_jockey_ps = {}
        self.old_trainer_ps = {}

    def load_data(self, start_date, end_date):
        races = self.extract_races(start_date, end_date)

        all_x = {}
        all_y = {}
        horse_nums = {}
        top_3 = {}

        for race in tqdm(races, desc="Loading data"):
            # remove inexperienced horses
            experienced_ps = get_experienced_ps(race)
            if experienced_ps is None:
                # if not enough experienced horses, remove
                continue
            m = len(experienced_ps)

            this_top_3 = []
            this_x = np.zeros((m, self.size), dtype=np.float32)
            this_y = np.zeros((m, 2), dtype=np.float32)
            this_horse_nums = np.zeros(m, dtype=int)

            for i, p in enumerate(experienced_ps):
                p_ranking = get_ranking_from_participation(p)
                this_horse_nums[i] = p.number
                this_y[i, 0] = p_ranking
                this_y[i, 1] = p.win_odds
                this_x[i] = self.load_participation(p)

                if p_ranking <= 3:
                    this_top_3.append((p.number, p_ranking))

            this_top_3.sort(key=lambda x: x[1])
            this_top_3 = np.array(this_top_3, dtype=np.int8)
            top_3[race.id] = this_top_3[:, 0]
            all_x[race.id] = this_x
            all_y[race.id] = this_y
            horse_nums[race.id] = this_horse_nums

        return all_x, all_y, horse_nums, top_3

    def load_participation(self, p):
        # current race rating stats
        mean_rating, std_rating, max_rating, min_rating = self.get_ratings_of_race(p.race)
        n = self.get_race_participants(p.race)

        # get historic races
        old_horse_ps = self.get_old_horse_ps(p.horse_id, p.race.date)
        old_jockey_ps = self.get_old_jockey_ps(p.jockey_id, p.race.date)[:50]
        # TODO: factor out magic number
        # assume len(trainer_ps) > 0 (not necessarily true)
        old_trainer_ps = self.get_old_trainer_ps(p.horse.trainer_id, p.race.date)[:300]

        # extract most recent race
        previous_horse_p = old_horse_ps[-1]
        previous_jockey_p = old_jockey_ps[-1]

        # extract meaningful data from ps
        h_days, h_diffs, h_speeds, h_ratings, h_rankings = self.extract_info_from_ps(p.race.date, old_horse_ps)
        j_days, j_diffs, j_speeds, _, j_rankings = self.extract_info_from_ps(p.race.date, old_jockey_ps)
        t_days, t_diffs, t_speeds, t_ratings, t_rankings = self.extract_info_from_ps(p.race.date, old_trainer_ps)

        h_rating_hist = np.insert(h_ratings, len(h_ratings), p.rating)
        weighted_rating_diffs = calculate_weighted_rating(h_days, h_diffs, h_rating_hist)

        return np.array([
            # individual participation features (8)
            p.rating,
            p.number,
            p.lane,
            p.horse_weight,
            p.gear_weight,
            p.gear_weight / (p.gear_weight + p.horse_weight),

            # race dependent features (6)
            # p.rating - mean_rating,
            # (p.rating - mean_rating) / std_rating,
            # max_rating - p.rating,
            # p.rating - min_rating,
            # p.lane / n,
            # p.number / n,

            # horse historic data (16)
            len(old_horse_ps),
            (p.race.date - previous_horse_p.race.date).days,
            get_ranking_from_participation(previous_horse_p),
            calculate_normalized_speed(previous_horse_p),
            get_maximum_speed(old_horse_ps),
            get_beaten_time(previous_horse_p),
            # capture more beaten_time historic data (mean and weighted mean)
            np.mean(h_speeds),
            np.std(h_speeds),
            calculate_weighted_mean_speed(h_days, h_speeds),
            previous_horse_p.win_odds,
            p.rating - h_ratings[-1],
            calculate_weighted_ranking(h_days, h_diffs, h_rankings, mode=WIN),
            calculate_weighted_ranking(h_days, h_diffs, h_rankings, mode=PLACE),
            calculate_weighted_ranking(h_days, h_diffs, h_rankings),
            np.mean(weighted_rating_diffs),
            np.std(weighted_rating_diffs),

            # jockey historic data (12)
            len(old_jockey_ps),
            get_ranking_from_participation(previous_jockey_p),
            calculate_normalized_speed(previous_jockey_p),
            get_maximum_speed(old_jockey_ps),
            get_beaten_time(previous_jockey_p),
            np.mean(j_speeds),
            np.std(j_speeds),
            calculate_weighted_mean_speed(j_days, j_speeds),
            previous_jockey_p.win_odds,
            calculate_weighted_ranking(j_days, j_diffs, j_rankings, mode=WIN),
            calculate_weighted_ranking(j_days, j_diffs, j_rankings, mode=PLACE),
            calculate_weighted_ranking(j_days, j_diffs, j_rankings),

            # trainer historic data (8)
            len(p.horse.trainer.horses),
            np.mean(t_speeds),
            np.std(t_speeds),
            calculate_weighted_mean_speed(t_days, t_speeds),
            np.mean(t_ratings),
            calculate_weighted_ranking(t_days, t_diffs, t_rankings, mode=WIN),
            calculate_weighted_ranking(t_days, t_diffs, t_rankings, mode=PLACE),
            calculate_weighted_ranking(t_days, t_diffs, t_rankings),
        ], dtype=np.float32)

    def extract_info_from_ps(self, race_date, ps):
        date_diffs = []
        speeds = []
        ratings = []
        difficulties = []
        rankings = []
        for p in ps:
            date_diffs.append((race_date - p.race.date).days)
            difficulties.append(self.get_race_difficulty(p.race))
            speeds.append(calculate_normalized_speed(p))
            ratings.append(p.rating if p.rating is not None else 25)
            rankings.append(get_ranking_from_participation(p))

        return (
            np.array(date_diffs, dtype=np.float32),
            np.array(difficulties, dtype=np.float32),
            np.array(speeds, dtype=np.float32),
            np.array(ratings, dtype=np.float32),
            np.array(rankings, dtype=np.float32)
        )

    def get_old_horse_ps(self, horse_id, before):
        session = self.session
        ps = (session.query(Participation)
              .join(Race)
              .filter(Participation.horse_id == horse_id).filter(Race.date < before)
              .order_by(Race.date)
              .all())
        return utils.remove_unranked_participants(ps)

    def get_old_jockey_ps(self, jockey_id, before):
        if (jockey_id, before) in self.old_jockey_ps:
            return self.old_jockey_ps[(jockey_id, before)]

        session = self.session
        ps = (session.query(Participation)
              .join(Race)
              .filter(Participation.jockey_id == jockey_id)
              .filter(Race.date < before)
              .order_by(Race.date)
              .all())
        result = utils.remove_unranked_participants(ps)
        self.old_jockey_ps[(jockey_id, before)] = result.copy()
        return result

    def get_old_trainer_ps(self, trainer_id, before):
        if (trainer_id, before) in self.old_trainer_ps:
            return self.old_trainer_ps[(trainer_id, before)]

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
        self.old_trainer_ps[(trainer_id, before)] = result.copy()
        return result

    def get_race_participants(self, race):
        if race.id not in self.race_participants:
            result = get_number_of_participants(race)
            self.race_participants[race.id] = result
        return self.race_participants[race.id]

    def get_ratings_of_race(self, race):
        if race.id not in self.race_ratings:
            mean, std = get_average_rating(race)
            max_rating = get_max_rating(race)
            min_rating = get_min_rating(race)
            self.race_ratings[race.id] = (mean, std, max_rating, min_rating)
        return self.race_ratings[race.id]

    def get_race_difficulty(self, race):
        if race.id not in self.race_difficulties:
            difficulty = calculate_race_difficulty(race)
            self.race_difficulties[race.id] = difficulty
        return self.race_difficulties[race.id]

    def get_race_ps(self, race):
        if race.id not in self.race_ps:
            result = utils.remove_unranked_participants(race.participations)
            self.race_ps[race.id] = result.copy()
        return self.race_ps[race.id]

    def extract_races(self, start_date, end_date):
        session = self.session
        races = session.query(Race).filter(Race.date >= start_date).filter(Race.date < end_date)
        if self.distance is not None:
            races = races.filter(Race.distance == self.distance)

        races = races.all()
        return races

    def close(self):
        self.session.close()


def main():
    init_engine()
    loader = DataLoader(size=42, distance=1400)

    data_x, data_y, horse_nums, top_3 = loader.load_data(
        start_date=datetime(2012,9,1).date(),
        end_date=datetime(2022,9,1).date(),
    )
    test_data_x, _, test_horse_nums, test_top_3 = loader.load_data(
        start_date=datetime(2022,9,1).date(),
        end_date=datetime(2024,9,1).date(),
    )

    final_data_x, _, final_horse_nums, final_top_3 = loader.load_data(
        start_date=datetime(2024,9,1).date(),
        end_date=datetime(2025,1,1).date(),
    )

    loader.close()

    directory = "data2"

    os.makedirs(directory, exist_ok=True)
    np.savez(f"{directory}/data_x.npz", **data_x)
    np.savez(f"{directory}/data_y.npz", **data_y)
    np.savez(f"{directory}/horse_nums.npz", **horse_nums)
    np.savez(f"{directory}/top_3.npz", **top_3)

    np.savez(f"{directory}/test_data_x.npz", **test_data_x)
    np.savez(f"{directory}/test_horse_nums.npz", **test_horse_nums)
    np.savez(f"{directory}/test_top_3.npz", **test_top_3)

    np.savez(f"{directory}/final_data_x.npz", **final_data_x)
    np.savez(f"{directory}/final_horse_nums.npz", **final_horse_nums)
    np.savez(f"{directory}/final_top_3.npz", **final_top_3)


if __name__ == '__main__':
    main()
