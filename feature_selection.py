import torch

import utils.utils as utils
from dataloader.utils import get_ranking_from_participation, calculate_speed, time_to_number_of_seconds
import numpy as np
import random

from utils.pools import *


# used data from 1/9/2010 to 29/12/2024
AVERAGE_SPEEDS = {
    1000: 17.34037,
    1200: 17.03160,
    1400: 16.83563,
    1600: 16.44470,
    1800: 16.37236,
    2000: 16.10338
}

def get_train_cv_split(data_keys, cv_ratio=0.1):
    m = len(data_keys)
    train_size = round(m * (1 - cv_ratio))
    train_keys = set(random.sample(data_keys, train_size))
    cv_keys = set(data_keys) - set(train_keys)
    return list(train_keys), list(cv_keys)


def is_new_horse(p):
    for this_p in p.horse.participations:
        if this_p.finish_time is not None and this_p.race.date < p.race.date:
            return False
    return True

def is_new_jockey(p):
    for this_p in p.jockey.participations:
        if this_p.finish_time is not None and this_p.race.date < p.race.date:
            return False
    return True


def count_new_horses(r):
    r_ps = utils.remove_unranked_participants(r.participations)
    counter = 0
    for p in r_ps:
        if is_new_horse(p):
            counter += 1
    return counter


def convert_ranking_to_score(ranking):
    return np.maximum((ranking <= 3) * (1.2 - 0.2 * ranking) + (ranking > 3) * (1.05 - 1.15 * ranking), 0)


def count_number_of_previous_races(p):
    h_ps = [hp for hp in utils.remove_unranked_participants(p.horse.participations) if hp.race.date < p.race.date]
    return len(h_ps)


def get_latest_horse_p(p):
    h_ps = [hp for hp in utils.remove_unranked_participants(p.horse.participations) if hp.race.date < p.race.date]
    return max(h_ps, key=lambda x: p.race.date)


def get_old_ps(p):
    h_ps = [hp for hp in utils.remove_unranked_participants(p.horse.participations) if hp.race.date < p.race.date]
    h_ps.sort(key=lambda x: p.race.date)
    return h_ps


def get_latest_jockey_p(p):
    j_ps = [jp for jp in utils.remove_unranked_participants(p.jockey.participations) if jp.race.date < p.race.date]
    return max(j_ps, key=lambda x: p.race.date)


def get_old_jockey_ps(p):
    j_ps = [jp for jp in utils.remove_unranked_participants(p.jockey.participations) if jp.race.date < p.race.date]
    j_ps.sort(key=lambda x: p.race.date)
    return j_ps


def get_old_trainer_ps(p):
    trainer = p.horse.trainer
    result = []
    for horse in trainer.horses:
        result += [hp for hp in utils.remove_unranked_participants(horse.participations) if hp.race.date < p.race.date]
    result.sort(key=lambda x: p.race.date)
    return result


def get_beaten_time(p):
    this_finish_time = p.finish_time
    race_ps = utils.remove_unranked_participants(p.race.participations)
    winner_time = min(race_ps, key=lambda x: x.finish_time).finish_time
    difference = time_to_number_of_seconds(this_finish_time) - time_to_number_of_seconds(winner_time)
    return difference


def get_relevant_ps_from_race(race):
    race_ps = utils.remove_unranked_participants(race.participations)
    result = [p for p in race_ps if not is_new_horse(p) and not is_new_jockey(p) and p.rating is not None]
    result.sort(key=lambda x: x.number)
    return result


def get_max_rating(race):
    ratings = [p.rating for p in get_relevant_ps_from_race(race)]
    return np.max(ratings)


def get_average_rating(race):
    ratings = [p.rating for p in get_relevant_ps_from_race(race)]
    return np.mean(ratings), np.std(ratings)


def get_min_rating(race):
    ratings = [p.rating for p in get_relevant_ps_from_race(race)]
    return np.min(ratings)


def get_min_max_rating(race):
    ratings = [p.rating for p in get_relevant_ps_from_race(race)]
    return min(ratings), max(ratings)


def calculate_defeated_rating(p):
    # sum of rating of horses it defeated - sum of rating of horses it got defeated by
    ranking = get_ranking_from_participation(p)
    race_ps = utils.remove_unranked_participants(p.race.participations)
    defeated = [op.rating if op.rating is not None else 25 for op in race_ps if get_ranking_from_participation(op) > ranking]
    lost_to  = [op.rating if op.rating is not None else 25 for op in race_ps if get_ranking_from_participation(op) < ranking]
    return (sum(defeated) - sum(lost_to)) / (len(defeated) + len(lost_to))


def calculate_mean_speed(ps):
    return np.mean(list(map(calculate_speed, ps)))


def calculate_std_speed(ps):
    return np.std(list(map(calculate_speed, ps)))


def get_maximum_speed(ps):
    return max(map(calculate_speed, ps))


def convert_distance_key(key):
    if key == 1650:
        return 1600
    elif key >= 2000:
        return 2000
    else:
        return key


def calculate_normalized_speed(p):
    key = p.race.distance
    key = convert_distance_key(key)
    average_speed = AVERAGE_SPEEDS[key]
    absolute_speed = calculate_speed(p)
    return absolute_speed - average_speed


def calculate_race_difficulty(race, rating_weight=0.7, number_weight=0.3):
    ps = get_relevant_ps_from_race(race)
    opponent_ratings = [this_p.rating for this_p in ps if this_p.rating is not None]
    rating_mean = np.mean(opponent_ratings)
    normalized_number = len(ps) / 14
    return rating_weight * rating_mean + number_weight * normalized_number


def get_weighted_ranking_rate(p, ps, diff_weight=0.7, date_weight=0.3, time_decay=0.03):
    race_date = p.race.date
    rankings = []
    dates_diff = []
    difficulty = []
    for this_p in ps:
        rankings.append(get_ranking_from_participation(this_p))
        dates_diff.append((race_date - this_p.race.date).days)
        difficulty.append(calculate_race_difficulty(this_p.race))

    rankings = torch.tensor(rankings, dtype=torch.float32, device="cuda")
    dates_diff = torch.tensor(dates_diff, dtype=torch.float32, device="cuda")
    dates_diff = torch.exp(-time_decay * dates_diff)
    difficulty = torch.tensor(difficulty, dtype=torch.float32, device="cuda")
    weights = dates_diff * date_weight + difficulty * diff_weight
    return weights.cpu().numpy(), rankings.cpu().numpy()


def calculate_weighted_ranking(weight, rankings, mode=None):
    if mode == WIN:
        rankings = (rankings == 1).astype(np.float32)
    elif mode == PLACE:
        rankings = (rankings <= 3).astype(np.float32)
    else:
        rankings = (14 - rankings) / 14

    return np.mean(weight * rankings)

def get_mean_rating(ps):
    ratings = [p.rating for p in ps if p.rating is not None]
    return np.mean(ratings)


def get_rating_diff(p, ps, diff_weight=0.7, date_weight=0.3, time_decay=0.03):
    # assume sorted in ascending order of date
    race_date = p.race.date
    all_ps = ps + [p]
    differences = []
    dates_diff = []
    difficulty = []

    for i in range(len(ps)):
        new = all_ps[i + 1]
        old = all_ps[i]
        differences.append(new.rating - (old.rating if old.rating is not None else 25))
        dates_diff.append((race_date - old.race.date).days)
        difficulty.append(calculate_race_difficulty(old.race))

    dates_diff = np.array(dates_diff, dtype=np.float32)
    dates_diff = np.exp(-time_decay * dates_diff)
    difficulty = np.array(difficulty, dtype=np.float32)
    differences = np.array(differences, dtype=np.float32)
    weights = dates_diff * date_weight + difficulty * diff_weight
    return weights * differences


# def calculate_mean_decay_speed(p, ps, time_decay=0.03):
#     race_date = p.race.date
#     corresponding = [[(race_date - this_p.race.date).days, calculate_speed(this_p)] for this_p in ps]
#     data = np.array(corresponding, dtype=np.float32)
#     days = data[:, 0]
#     speeds = data[:, 1]
#     weighting_factor = np.exp(-time_decay * days)
#     return np.dot(weighting_factor, speeds) / np.sum(weighting_factor)
#
# def calculate_weighted_mean_speed(p, ps, time_decay=0.03):
#     # Create a list of (time_diff, speed) pairs
#     race_date = p.race.date
#     corresponding = [
#         [(race_date - this_p.race.date).days, calculate_speed(this_p)]
#         for this_p in ps
#     ]
#
#     # Convert to a numpy array for easier manipulation
#     data = np.array(corresponding, dtype=np.float32)
#     days = data[:, 0]  # Time difference in days
#     speeds = data[:, 1]  # Speed values
#
#     # Calculate weights based on exponential decay
#     weighting_factor = np.exp(-time_decay * days)
#
#     # Normalize the weights so they sum up to 1
#     normalized_weights = weighting_factor / np.sum(weighting_factor)
#
#     # Compute weighted mean of speeds
#     weighted_mean_speed = np.dot(normalized_weights, speeds)
#
#     return weighted_mean_speed



# if __name__ == '__main__':
#     from database import Race, init_engine, get_session
#     from tqdm import tqdm
#     init_engine()
#     session = get_session()
#     races = session.query(Race).all()
#     xs = []
#     for race in tqdm(races):
#         xs.append(calculate_race_difficulty(race))
#     print(np.mean(xs))
