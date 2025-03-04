# 64 PERFECT FEATURES - NO TURNING BACK. LAST FILE ON DATA LOADING.
import numpy as np

from database import Race, Participation, Horse, Trainer, init_engine, get_session
from datetime import time

from utils import utils
from utils.pools import WIN, PLACE

from functools import lru_cache
from tqdm import tqdm


def get_ranking_from_participation(x):
    return int(x.ranking.replace("DH", "").strip())


def get_winner(race: Race) -> Participation:
    # if multiple, return first one (rare case)
    return min(race.participations, key=lambda p: get_ranking_from_participation(p))


def get_seconds_from_time(t: time) -> float:
    return t.minute * 60 + t.second + t.microsecond * 1e-6


def get_speed(p: Participation) -> float:
    # time in m/s
    distance = p.race.distance
    seconds = get_seconds_from_time(p.finish_time)
    return distance / seconds


def get_adjusted_beaten_time(p: Participation) -> float:
    winner = get_winner(p.race)
    winner_finish_time = get_seconds_from_time(winner.finish_time)
    this_finish_time = get_seconds_from_time(p.finish_time)

    # TODO: think about how distance and stuff is affected (call get_speed_factor??)
    raise NotImplementedError


@lru_cache(maxsize=None)
def get_adjusted_ranking(ranking, total):
    return (ranking - 1) / (total - 1)


@lru_cache(maxsize=None)
def get_score_from_ranking(ranking, decay_sharpness=2):
    return 1 / np.log(ranking + decay_sharpness)


@lru_cache(maxsize=None)
def get_adjusted_win_odds(win_odds, total):
    raise NotImplementedError


@lru_cache(maxsize=None)
def get_race_difficulty(race: Race) -> float:
    raise NotImplementedError


@lru_cache(maxsize=None)
def get_race_relevance(race: Race, race_date, race_condition, race_location, jockey_id, trainer_id) -> float:
    raise NotImplementedError


@lru_cache(maxsize=None)
def get_speed_factor(distance, condition) -> float:
    # TODO: come up with equation
    raise NotImplementedError


@lru_cache(maxsize=None)
def encode_condition(condition):
    raise NotImplementedError


@lru_cache(maxsize=None)
def encode_location(location):
    raise NotImplementedError


@lru_cache(maxsize=None)
def get_time_relevancy(current_date, old_date, time_decay=0.03) -> float:
    days_diff = (current_date - old_date).days
    return np.exp(-time_decay * days_diff)


@lru_cache(maxsize=None)
def get_total_participants(race):
    ps = race.participations
    return len(utils.remove_unranked_participants(ps))


def get_adjusted_speed(p: Participation) -> float:
    raw_speed = get_speed(p)

    distance = p.race.distance
    condition = p.race.condition

    factor = get_speed_factor(distance, condition)
    return raw_speed * factor


def get_equivalent_distance(distance):
    if distance == 1600:
        return 1650
    elif distance > 2000:
        return 2000
    else:
        return distance


def extract_races(session, start_date, end_date, distance):
    races = session.query(Race).filter(Race.date >= start_date).filter(Race.date < end_date)
    if distance is not None:
        equivalent_distance = get_equivalent_distance(distance)
        races = races.filter(Race.distance == equivalent_distance)
    races = races.all()
    return races


def get_experienced_ps(race):
    race_ps = utils.remove_unranked_participants(race.participations)
    result = [p for p in race_ps if not utils.is_new_horse(p) and not utils.is_new_jockey(p) and p.rating is not None]
    result.sort(key=lambda x: x.number)
    valid_count = len(result)
    if valid_count >= 4:
        return result
    else:
        return None


def is_place(ranking, participants):
    return ranking <= 2 | (ranking == 3 & participants >= 7)


def get_old_horse_ps(session, horse_id, before):
    ps = (session.query(Participation)
          .join(Race)
          .filter(Participation.horse_id == horse_id).filter(Race.date < before)
          .order_by(Race.date)
          .all())
    return utils.remove_unranked_participants(ps)


def get_old_jockey_ps(session, jockey_id, before):
    ps = (session.query(Participation)
          .join(Race)
          .filter(Participation.jockey_id == jockey_id).filter(Race.date < before)
          .order_by(Race.date)
          .all())
    return utils.remove_unranked_participants(ps)


def get_old_trainer_ps(session, trainer_id, before):
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


def extract_info_from_ps(race_date, ps):
    n = len(ps)
    result = np.zeros((n, 11), dtype=np.float64)

    for i, p in enumerate(ps):
        participants = get_total_participants(p.race)
        result[i, 0] = get_time_relevancy(race_date, p.race.date)
        result[i, 1] = p.race.distance
        result[i, 2] = encode_location(p.race.location)
        result[i, 3] = encode_condition(p.race.condition)
        result[i, 4] = get_race_difficulty(p.race)
        result[i, 5] = participants
        result[i, 6] = get_adjusted_speed(p)
        result[i, 7] = get_ranking_from_participation(p)
        result[i, 8] = p.rating
        result[i, 9] = get_adjusted_win_odds(p.win_odds, participants)
        result[i, 10] = get_adjusted_beaten_time(p)

    return result


def normalize_weights(*weights):
    total = sum(weights)
    result = []
    for w in weights:
        result.append(w / total)
    return result


def get_weighted_speed(time_relevancy, track_relevancy, speeds, time_weighting=0.7, track_weighting=0.3):
    time_weighting, track_weighting = normalize_weights(time_weighting, track_weighting)
    total_weights = time_weighting * time_relevancy + track_weighting * track_relevancy
    return np.dot(total_weights, speeds) / np.sum(total_weights)


def get_weighted_ranking(
    time_relevancy,
    track_relevancy,
    race_difficulty,
    ranking,
    participants,
    mode=None,
    time_weight=0.3,
    track_weight=0.2,
    difficulty_weight=0.5
):
    time_weight, track_weight, difficulty_weight = normalize_weights(time_weight, track_weight, difficulty_weight)
    total_weights = time_weight * time_relevancy + track_weight * track_relevancy + difficulty_weight * race_difficulty

    if mode == WIN:
        ranking = (ranking == 1).astype(np.float64)
        score = ranking
    elif mode == PLACE:
        ranking = is_place(ranking, participants).astype(np.float64)
        score = ranking
    else:
        ranking = get_adjusted_ranking(ranking, participants)
        score = get_score_from_ranking(ranking)

    return np.dot(score, total_weights) / np.sum(total_weights)


def get_weighted_delta_rating(
    time_relevancy,
    track_relevancy,
    race_difficulty,
    rating_hist,
    current_rating,
    time_weight=0.3,
    track_weight=0.2,
    difficulty_weight=0.5,
):
    raise NotImplementedError("both +ve and -ve rating changes should scale in respective ways")
    time_weight, track_weight, difficulty_weight = normalize_weights(time_weight, track_weight, difficulty_weight)

    n = rating_hist.shape[0]
    rating_diff = np.zeros(n, dtype=np.float64)
    rating_diff[:n - 1] = np.diff(rating_hist)
    rating_diff[n - 1] = current_rating - rating_hist[-1]

    total_weights = time_weight * time_relevancy + track_weight * track_relevancy + difficulty_weight * race_difficulty
    return np.dot(total_weights, rating_diff) / np.sum(total_weights)


def get_weighted_beaten_length(
    time_relevancy,
    track_relevancy,
    race_difficulty,
    beaten_length,
    time_weight=0.3,
    track_weight=0.2,
    difficulty_weight=0.5,
):
    time_weight, track_weight, difficulty_weight = normalize_weights(time_weight, track_weight, difficulty_weight)
    total_weight = time_weight * time_relevancy + track_weight * track_relevancy + difficulty_weight * race_difficulty

    return np.dot(total_weight, beaten_length) / np.sum(total_weight)


def get_track_relevancy(
    race_distance,
    race_location,
    race_condition,
    distance,
    location,
    condition,
    distance_weight=0.7,
    location_weight=0.1,
    condition_weight=0.2,
):
    distance_weight, location_weight, condition_weight = normalize_weights(distance_weight, location_weight, condition_weight)
    raise NotImplementedError


def scale_by_difficulty(difficulty, variable):
    n = difficulty.shape[0]
    return difficulty * variable * n / np.sum(difficulty)


def weigh_by_relevancy_mean(time_relevancy, track_relevancy, variable, time_weight=0.7, track_weight=0.3):
    time_weight, track_weight = normalize_weights(time_weight, track_weight)
    weight = time_weight * time_relevancy + track_weight * track_relevancy

    return np.dot(weight, variable) / np.sum(weight)


def weigh_by_relevancy_std(time_relevancy, track_relevancy, variable, weighted_mean, time_weight=0.7, track_weight=0.3):
    time_weight, track_weight = normalize_weights(time_weight, track_weight)
    weight = time_weight * time_relevancy + track_weight * track_relevancy

    return np.sqrt(weight * np.square(variable - weighted_mean) / np.sum(weight))


def load_data(session, size, start_date, end_date, distance):
    races = extract_races(session, start_date, end_date, distance)

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
            this_y[i, 1] = get_adjusted_ranking(p_ranking, participants)
            this_y[i, 2] = get_seconds_from_time(p.finish_time)
            this_y[i, 3] = get_speed(p)
            this_y[i, 4] = p.win_odds

            this_x[i] = load_p(session, p)

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


def load_p(session, p: Participation):
    trainer = p.horse.trainer

    old_horse_ps = get_old_horse_ps(session, p.horse_id, p.race.date)
    old_jockey_ps = get_old_jockey_ps(session, p.jockey_id, p.race.date)[-50:]
    old_trainer_ps = get_old_trainer_ps(session, p.horse.trainer_id, p.race.date)[-300:]

    previous_horse_p = old_horse_ps[-1]
    previous_jockey_p = old_jockey_ps[-1]

    horse_ps_info = extract_info_from_ps(p.race.date, old_horse_ps)
    jockey_ps_info = extract_info_from_ps(p.race.date, old_jockey_ps)
    trainer_ps_info = extract_info_from_ps(p.race.date, old_trainer_ps)

    horse_time_relevancy = horse_ps_info[0]
    jockey_time_relevancy = jockey_ps_info[0]
    trainer_time_relevancy = trainer_ps_info[0]

    horse_track_relevancy = get_track_relevancy(p.race.distance, p.race.location, p.race.condition, horse_ps_info[1], horse_ps_info[2], horse_ps_info[3])
    jockey_track_relevancy = get_track_relevancy(p.race.distance, p.race.location, p.race.condtion, jockey_ps_info[1], jockey_ps_info[2], jockey_ps_info[3])
    trainer_track_relevancy = get_track_relevancy(p.race.distance, p.race.location, p.race.condition, trainer_ps_info[1], trainer_ps_info[2], trainer_ps_info[3])

    horse_race_difficulty = horse_ps_info[4]
    jockey_race_difficulty = jockey_ps_info[4]
    trainer_race_difficulty = trainer_ps_info[4]

    # Info is as follows:
    # 0. Time relevancy
    # 1. Distance
    # 2. Location
    # 3. Condition
    # 4. Race difficulty
    # 5. Race participants
    # 6. Adjusted speed
    # 7. Ranking
    # 8. Rating
    # 9. Adjusted win odds
    # 10. Adjusted beaten length


    # Speeds
    horse_mean_weighted_speed = weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_ps_info[6])
    horse_std_weighted_speed = weigh_by_relevancy_std(horse_time_relevancy, horse_track_relevancy, horse_ps_info[6], horse_mean_weighted_speed)

    jockey_mean_weighted_speed = weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, trainer_ps_info[6])
    jockey_std_weighted_speed = weigh_by_relevancy_std(jockey_time_relevancy, jockey_track_relevancy, trainer_ps_info[6], jockey_mean_weighted_speed)

    trainer_mean_weighted_speed = weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[6])
    trainer_std_weighted_speed = weigh_by_relevancy_std(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[6], trainer_mean_weighted_speed)

    # Rankings
    horse_score = get_score_from_ranking(get_adjusted_ranking(horse_ps_info[7], horse_ps_info[5]))
    horse_scaled_win_rate = scale_by_difficulty(horse_race_difficulty, (horse_ps_info[7] == 1).astype(np.float64))
    horse_scaled_place_rate = scale_by_difficulty(horse_race_difficulty, is_place(horse_ps_info[7], horse_ps_info[5]).astype(np.float64))
    horse_scaled_score = scale_by_difficulty(horse_race_difficulty, horse_score)
    horse_mean_weighted_win_rate = weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_win_rate)
    horse_mean_weighted_place_rate = weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_place_rate)
    horse_mean_weighted_score = weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_score)
    horse_std_weighted_score = weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_scaled_score, horse_mean_weighted_score)

    jockey_score = get_score_from_ranking(get_adjusted_ranking(jockey_ps_info[7], jockey_ps_info[5]))
    jockey_scaled_win_rate = scale_by_difficulty(jockey_race_difficulty, (jockey_ps_info[7] == 1).astype(np.float64))
    jockey_scaled_place_rate = scale_by_difficulty(jockey_race_difficulty, is_place(jockey_ps_info[7], jockey_ps_info[5]).astype(np.float64))
    jockey_scaled_score = scale_by_difficulty(jockey_race_difficulty, jockey_score)
    jockey_mean_weighted_win_rate = weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_win_rate)
    jockey_mean_weighted_place_rate = weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_place_rate)
    jockey_mean_weighted_score = weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_score)
    jockey_std_weighted_score = weigh_by_relevancy_mean(jockey_time_relevancy, jockey_track_relevancy, jockey_scaled_score, jockey_mean_weighted_score)

    trainer_score = get_score_from_ranking(get_adjusted_ranking(trainer_ps_info[7], trainer_ps_info[5]))
    trainer_scaled_win_rate = scale_by_difficulty(trainer_race_difficulty, (trainer_ps_info[7] == 1).astype(np.float64))
    trainer_scaled_place_rate = scale_by_difficulty(trainer_race_difficulty,is_place(trainer_ps_info[7], trainer_ps_info[5]).astype(np.float64))
    trainer_scaled_score = scale_by_difficulty(trainer_race_difficulty, trainer_score)
    trainer_mean_weighted_win_rate = weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy,trainer_scaled_win_rate)
    trainer_mean_weighted_place_rate = weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy,trainer_scaled_place_rate)
    trainer_mean_weighted_score = weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy,trainer_scaled_score)
    trainer_std_weighted_score = weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy,trainer_scaled_score, trainer_mean_weighted_score)

    # Ratings
    # a) horse should use both rating and rating diff
    # b) trainer should use just rating

    horse_mean_weighted_rating = weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_ps_info[8], track_weight=0)
    horse_std_weighted_rating = weigh_by_relevancy_mean(horse_time_relevancy, horse_track_relevancy, horse_ps_info[8], horse_mean_weighted_rating, track_weight=0)

    # TODO: delta rating

    trainer_mean_weighted_rating = weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[8], track_weight=0)
    trainer_std_weighted_rating = weigh_by_relevancy_mean(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[8], trainer_mean_weighted_rating, track_weight=0)

    # Beaten length


    # Win odds


    return np.array([
        # independent features (6)
        p.rating,
        p.number,
        p.lane,
        p.horse_weight,
        p.gear_weight,
        p.gear_weight / (p.horse_weight + p.horse_weight),

        # previous horse stats (4)
        get_adjusted_speed(previous_horse_p),
        p.rating - previous_horse_p.rating,
        (p.race.date - previous_horse_p.race.date).days,
        get_adjusted_beaten_time(previous_horse_p),

        # previous jockey stats (3)
        get_adjusted_speed(previous_jockey_p),
        (p.race.date - previous_jockey_p.race.date).days,
        get_adjusted_beaten_time(previous_jockey_p),

        # horse historic data (6)
        len(old_horse_ps),
        get_weighted_speed(horse_time_relevancy, horse_track_relevancy, horse_ps_info[6]),  # measure speed but with weighted mean
        np.std(horse_ps_info[6]),                                                           # measure consistency of speed
        get_weighted_ranking(horse_time_relevancy, horse_track_relevancy, horse_race_difficulty, horse_ps_info[7], horse_ps_info[5], mode=WIN),
        get_weighted_ranking(horse_time_relevancy, horse_track_relevancy, horse_race_difficulty, horse_ps_info[7], horse_ps_info[5], mode=PLACE),
        get_weighted_ranking(horse_time_relevancy, horse_track_relevancy, horse_race_difficulty, horse_ps_info[7], horse_ps_info[5]),
        np.std(get_adjusted_ranking(horse_ps_info[7], horse_ps_info[5])),    # ranking consistency
        get_weighted_delta_rating(horse_time_relevancy, horse_track_relevancy, horse_race_difficulty, horse_ps_info[8], p.rating),
        get_weighted_beaten_length(horse_time_relevancy, horse_track_relevancy, horse_race_difficulty, horse_ps_info[10]),
        # capture consistency of beaten length

        # jockey historic data (6)
        len(old_jockey_ps),
        get_weighted_speed(jockey_time_relevancy, jockey_track_relevancy, jockey_ps_info[6]),
        np.std(jockey_ps_info[6]),
        get_weighted_ranking(jockey_time_relevancy, jockey_track_relevancy, jockey_race_difficulty, jockey_ps_info[7], jockey_ps_info[5], mode=WIN),
        get_weighted_ranking(jockey_time_relevancy, jockey_track_relevancy, jockey_race_difficulty, jockey_ps_info[7], jockey_ps_info[5], mode=PLACE),
        get_weighted_ranking(jockey_time_relevancy, jockey_track_relevancy, jockey_race_difficulty, jockey_ps_info[7], jockey_ps_info[5]),
        np.std(get_adjusted_ranking(jockey_ps_info[7], jockey_ps_info[5])),   # ranking consistency
        get_weighted_beaten_length(jockey_time_relevancy, jockey_track_relevancy, jockey_race_difficulty, jockey_ps_info[10]),
        # capture consistency of beaten length

        # trainer historic data (8)
        len(trainer.horses),
        get_weighted_speed(trainer_time_relevancy, trainer_track_relevancy, trainer_ps_info[6]),
        np.std(trainer_ps_info[6]),
        get_weighted_ranking(trainer_time_relevancy, trainer_track_relevancy, trainer_race_difficulty, trainer_ps_info[7], trainer_ps_info[5], mode=WIN),
        get_weighted_ranking(trainer_time_relevancy, trainer_track_relevancy, trainer_race_difficulty, trainer_ps_info[7], trainer_ps_info[5], mode=PLACE),
        get_weighted_ranking(trainer_time_relevancy, trainer_track_relevancy, trainer_race_difficulty, trainer_ps_info[7], trainer_ps_info[5]),
        np.std(get_adjusted_ranking(trainer_ps_info[7], trainer_ps_info[5])),  # ranking consistency
        np.mean(trainer_ps_info[8]),
        np.std(trainer_ps_info[8]),

        # jockey x trainer combo

    ], dtype=np.float64)


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


def main():
    init_engine()
    session = get_session()

    session.close()
