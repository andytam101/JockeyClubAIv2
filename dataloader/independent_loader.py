from .config import *
from .utils import *
import numpy as np
from datetime import datetime, timedelta

from database import Horse, Jockey, Participation, Race

# data sizes
PARTICIPATION_FEATURES = 6
HORSE_FEATURES = 48
JOCKEY_FEATURES = 30
TRAINER_FEATURES = 19
HORSE_JOCKEY_FEATURES = 36
HORSE_DISTANCE_FEATURES = 33
JOCKEY_DISTANCE_FEATURES = 25
HORSE_JOCKEY_DISTANCE_FEATURES = 29
HORSE_TRACK_FEATURES = 33
HORSE_JOCKEY_TRACK_FEATURES = 29
HORSE_CONDITION_FEATURES = 33
HORSE_JOCKEY_CONDITION_FEATURES = 29
JOCKER_TRAINER_FEATURES = 16


INDEPENDENT_FEATURES = (
        PARTICIPATION_FEATURES +
        HORSE_FEATURES +
        JOCKEY_FEATURES +
        TRAINER_FEATURES +
        HORSE_JOCKEY_FEATURES +
        HORSE_DISTANCE_FEATURES +
        JOCKEY_DISTANCE_FEATURES +
        HORSE_JOCKEY_DISTANCE_FEATURES +
        HORSE_TRACK_FEATURES +
        HORSE_JOCKEY_TRACK_FEATURES +
        HORSE_CONDITION_FEATURES +
        HORSE_JOCKEY_CONDITION_FEATURES +
        JOCKER_TRAINER_FEATURES
    )  # total = 366


def get_combo_ps(p, ps, session, prediction, jockey=False, trainer=False, condition=False, distance=False, track=False):
    if not prediction:
        jockey_obj = p.jockey
        trainer_obj = p.horse.trainer

        if jockey:
            ps = list(filter(lambda x: x.jockey_id == jockey_obj.id, ps))
        if trainer:
            ps = list(filter(lambda x: x.horse.trainer_id == trainer_obj.id, ps))
        if condition:
            ps = list(filter(lambda x: x.race.condition == p.race.condition, ps))
        if distance:
            ps = list(filter(lambda x: x.race.distance == p.race.distance, ps))
        if track:
            ps = list(filter(lambda x: x.race.course == p.race.course, ps))
    else:
        if jockey:
            ps = list(filter(lambda x: x.jockey_id == p["jockey_id"], ps))
        if trainer:
            horse_obj = session.query(Horse).filter(Horse.id == p["horse_id"]).one()
            trainer_obj = horse_obj.trainer
            ps = list(filter(lambda x: x.horse.trainer_id == trainer_obj.id, ps))
        if condition:
            ps = list(filter(lambda x: x.race.condition == p["condition"], ps))
        if distance:
            ps = list(filter(lambda x: x.race.distance == p["distance"], ps))
        if track:
            ps = list(filter(lambda x: x.race.course == p["course"], ps))

    return ps


def get_general_group_data(p, ps, prediction=False):
    if not prediction:
        race_date = p.race.date
    else:
        race_date = p["date"]

    if len(ps) > 0:
        most_recent = ps[0]
        diff = race_date - most_recent.race.date
    else:
        diff = timedelta(days=COUNT_DAYS_BACKWARD)

    return {
        "count": len(ps),
        "days_since": diff.days
    }


def load_speed_data(ps):
    speeds = list(map(calculate_speed, ps))
    mean, std, latest = mean_std_latest(speeds)

    running_average = calculate_running_average(speeds, RUNNING_AVERAGE_COUNT)
    _, running_std, running_latest = mean_std_latest(running_average)

    speed_gradient = calculate_gradient(running_average)
    gradient_mean, _, gradient_latest = mean_std_latest(speed_gradient)

    return {
        "mean": mean,
        "std": std,
        "latest": latest,
        "running_average": running_average,
        "running_std": running_std,
        "running_latest": running_latest,
        "gradient_mean": gradient_mean,
        "gradient_latest": gradient_latest,
    }


def load_horse_weight_data(ps):
    horse_weights = list(map(lambda x: x.horse_weight, ps))
    mean, std, _ = mean_std_latest(horse_weights)

    running_average = calculate_running_average(horse_weights, RUNNING_AVERAGE_COUNT)
    _, running_std, running_latest = mean_std_latest(running_average)

    weight_gradient = calculate_gradient(running_average)
    gradient_mean, _, gradient_latest = mean_std_latest(weight_gradient)

    weight_diff = np.diff(horse_weights)
    diff_mean, diff_std, diff_latest = mean_std_latest(weight_diff)

    return {
        "mean": mean,
        "std": std,
        "running_std": running_std,
        "running_latest": running_latest,
        "gradient_mean": gradient_mean,
        "gradient_latest": gradient_latest,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_latest": diff_latest,
    }


def load_ranking_data(ps):
    if len(ps) == 0:
        return {
            "median": 0,
            "iqr": 0,
            "latest": 0,
            "normalized_mean": 0,
            "normalized_std": 0,
            "normalized_latest": 0,
        }
    rankings = list(map(get_ranking_from_participation, ps))
    normalized_rankings = list(map(calculate_normalized_rankings, ps))
    normalized_mean, normalized_std, normalized_latest = mean_std_latest(normalized_rankings)

    median = np.median(rankings)
    upper_quartile, lower_quartile = np.percentile(rankings, [75, 25])
    iqr = upper_quartile - lower_quartile
    latest = rankings[0]

    return {
        "median": median,
        "iqr": iqr,
        "latest": latest,
        "normalized_mean": normalized_mean,
        "normalized_std": normalized_std,
        "normalized_latest": normalized_latest,
    }


def load_rating_data(ps):
    rating = np.array(list(map(lambda x: x.rating, ps)), dtype=np.float32)
    rating[np.isnan(rating)] = 25

    mean, std, _ = mean_std_latest(rating)

    running_average = calculate_running_average(rating, RUNNING_AVERAGE_COUNT)
    _, running_std, running_latest = mean_std_latest(running_average)

    rating_gradient = calculate_gradient(running_average)
    gradient_mean, _, gradient_latest = mean_std_latest(rating_gradient)

    rating_diff = np.diff(rating).tolist()
    diff_mean, diff_std, diff_latest = mean_std_latest(rating_diff)

    return {
        "mean": mean,
        "std": std,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_latest": diff_latest,
        "running_std": running_std,
        "running_latest": running_latest,
        "gradient_mean": gradient_mean,
        "gradient_latest": gradient_latest,
    }


def load_win_odds_data(ps):
    win_odds = list(map(lambda x: x.win_odds, ps))
    mean, std, latest = mean_std_latest(win_odds)

    running_average = calculate_running_average(win_odds, RUNNING_AVERAGE_COUNT)
    _, running_std, running_latest = mean_std_latest(running_average)

    win_odds_gradient = calculate_gradient(running_average)
    gradient_mean, _, gradient_latest = mean_std_latest(win_odds_gradient)

    return {
        "mean": mean,
        "std": std,
        "latest": latest,
        "running_std": running_std,
        "running_latest": running_latest,
        "gradient_mean": gradient_mean,
        "gradient_latest": gradient_latest,
    }


def get_number_data(ps):
    count_ranking = lambda x: len(list(filter(lambda p: get_ranking_from_participation(p) <= x, ps)))
    return {
        "top_1_number": count_ranking(1),
        "top_2_number": count_ranking(2),
        "top_3_number": count_ranking(3),
        "top_4_number": count_ranking(4),
    }


def get_ratio_data(ps):
    if len(ps) == 0:
        return {
            "top_1_ratio": 0,
            "top_2_ratio": 0,
            "top_3_ratio": 0,
            "top_4_ratio": 0,
        }
    calculate_ratio = lambda x: len(list(filter(lambda p: get_ranking_from_participation(p) <= x, ps))) / len(ps)
    return {
        "top_1_ratio": calculate_ratio(1),
        "top_2_ratio": calculate_ratio(2),
        "top_3_ratio": calculate_ratio(3),
        "top_4_ratio": calculate_ratio(4),
    }


def load_participation_features(p):
    # length = 6
    number_of_participants = get_number_of_participants(p.race)
    return [25 if p.rating is None else p.rating, p.number, p.lane, p.lane / number_of_participants, p.horse_weight, p.gear_weight]


def load_predict_participation_features(p, number_of_participants):
    return [p["rating"], p["number"], p["lane"], p["lane"] / number_of_participants, p["horse_weight"], p["gear_weight"]]


def get_horse_ps_from_participation(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    return ps, ps_days


def get_jockey_ps_from_participation(p):
    jockey = p.jockey
    ps = jockey.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    return ps, ps_days


def get_trainer_ps_from_horse(horse):
    trainer = horse.trainer
    horses = trainer.horses
    result = []
    for horse in horses:
        result += horse.participations
    return result


def get_trainer_ps_from_participation(p):
    horse = p.horse
    ps = get_trainer_ps_from_horse(horse)
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date,
                                             start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    return ps, ps_days


def get_horse_ps_from_dictionary(session, p):
    horse_id = p["horse_id"]
    horse_obj = session.query(Horse).filter(Horse.id == horse_id).one_or_none()

    if horse_obj is None:
        # TODO: consider problem when horse is not in database
        return [], []

    ps = horse_obj.participations
    race_date = p["date"]
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    return  ps, ps_days


def get_jockey_ps_from_dictionary(session, p):
    jockey_id = p["jockey_id"]
    jockey_obj = session.query(Jockey).filter(Jockey.id == jockey_id).one_or_none()

    if jockey_obj is None:
        # TODO: consider problem when horse is not in database
        return [], []

    ps = jockey_obj.participations
    race_date = p["date"]
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))

    return ps, ps_days


def get_trainer_ps_from_dictionary(session, p: dict):
    horse_id = p["horse_id"]
    horse = session.query(Horse).filter(Horse.id == horse_id).one_or_none()
    if horse is None:
        return [], []

    ps = get_trainer_ps_from_horse(horse)
    race_date = p["date"]
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date,
                                             start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    return ps, ps_days


def load_horse_features(p, session, prediction):
    # length = 48
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    horse_weight_data = load_horse_weight_data(ps)
    ranking_data = load_ranking_data(ps)
    rating_data = load_rating_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        general_group_data["days_since"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        rating_data["mean"],
        rating_data["std"],
        rating_data["diff_mean"],
        rating_data["diff_std"],
        rating_data["diff_latest"],
        rating_data["running_std"],
        rating_data["running_latest"],
        rating_data["gradient_mean"],
        rating_data["gradient_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        horse_weight_data["mean"],
        horse_weight_data["std"],
        horse_weight_data["diff_mean"],
        horse_weight_data["diff_std"],
        horse_weight_data["diff_latest"],
        horse_weight_data["running_std"],
        horse_weight_data["running_latest"],
        horse_weight_data["gradient_mean"],
        horse_weight_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def load_jockey_features(p, session, prediction):
    # length = 30
    if not prediction:
        ps, ps_days = get_jockey_ps_from_participation(p)
    else:
        ps, ps_days = get_jockey_ps_from_dictionary(session, p)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        general_group_data["days_since"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def load_trainer_features(p, session, prediction):
    # length = 19
    if not prediction:
        ps, ps_days = get_trainer_ps_from_participation(p)
    else:
        ps, ps_days = get_trainer_ps_from_dictionary(session, p)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    win_odds_data = load_win_odds_data(ps)
    ranking_data = load_ranking_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_horse_jockey_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = get_combo_ps(p, ps, session, prediction, jockey=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    rating_data = load_rating_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        rating_data["diff_mean"],
        rating_data["diff_std"],
        rating_data["diff_latest"],
        rating_data["running_std"],
        rating_data["running_latest"],
        rating_data["gradient_mean"],
        rating_data["gradient_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_horse_distance_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = get_combo_ps(p, ps, session, prediction, distance=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    rating_data = load_rating_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        rating_data["running_std"],
        rating_data["running_latest"],
        rating_data["gradient_mean"],
        rating_data["gradient_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_jockey_distance_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_jockey_ps_from_participation(p)
    else:
        ps, ps_days = get_jockey_ps_from_dictionary(session, p)

    ps = get_combo_ps(p, ps, session, prediction, distance=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_horse_jockey_distance_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)

    ps = get_combo_ps(p, ps, session, prediction, distance=True, jockey=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_horse_track_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = get_combo_ps(p, ps, session, prediction, track=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    rating_data = load_rating_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        rating_data["running_std"],
        rating_data["running_latest"],
        rating_data["gradient_mean"],
        rating_data["gradient_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_horse_jockey_track_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = get_combo_ps(p, ps, session, prediction, jockey=True, track=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_horse_condition_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = get_combo_ps(p, ps, session, prediction, track=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    rating_data = load_rating_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        rating_data["running_std"],
        rating_data["running_latest"],
        rating_data["gradient_mean"],
        rating_data["gradient_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_horse_jockey_condition_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = get_combo_ps(p, ps, session, prediction, jockey=True, condition=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    win_odds_data = load_win_odds_data(ps)
    top_number_data = get_number_data(ps)
    top_ratio_data = get_ratio_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        speed_data["running_std"],
        speed_data["running_latest"],
        speed_data["gradient_mean"],
        speed_data["gradient_latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
        win_odds_data["running_std"],
        win_odds_data["running_latest"],
        win_odds_data["gradient_mean"],
        win_odds_data["gradient_latest"],
        top_number_data["top_1_number"],
        top_number_data["top_2_number"],
        top_number_data["top_3_number"],
        top_number_data["top_4_number"],
        top_ratio_data["top_1_ratio"],
        top_ratio_data["top_2_ratio"],
        top_ratio_data["top_3_ratio"],
        top_ratio_data["top_4_ratio"],
    ]


def get_jockey_trainer_features(p, session, prediction):
    if not prediction:
        ps, ps_days = get_horse_ps_from_participation(p)
    else:
        ps, ps_days = get_horse_ps_from_dictionary(session, p)
    ps = get_combo_ps(p, ps, session, prediction, trainer=True)
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days, prediction)
    speed_data = load_speed_data(ps)
    ranking_data = load_ranking_data(ps)
    rating_data = load_rating_data(ps)
    win_odds_data = load_win_odds_data(ps)

    return [
        general_group_data["count"],
        speed_data["mean"],
        speed_data["std"],
        speed_data["latest"],
        ranking_data["median"],
        ranking_data["iqr"],
        ranking_data["latest"],
        ranking_data["normalized_mean"],
        ranking_data["normalized_std"],
        ranking_data["normalized_latest"],
        rating_data["diff_mean"],
        rating_data["diff_std"],
        rating_data["diff_latest"],
        win_odds_data["mean"],
        win_odds_data["std"],
        win_odds_data["latest"],
    ]


def load_one_independent_participation(p, session, prediction, number_of_participants=None):
    assert not prediction or number_of_participants is not None

    if not prediction:
        p_features = load_participation_features(p)
    else:
        p_features = load_predict_participation_features(p, number_of_participants)
    horse_features = load_horse_features(p, session, prediction)
    jockey_features = load_jockey_features(p, session, prediction)
    trainer_features = load_trainer_features(p, session, prediction)
    h_j_features = get_horse_jockey_features(p, session, prediction)
    h_d_features = get_horse_distance_features(p, session, prediction)
    j_d_features = get_jockey_distance_features(p, session, prediction)
    h_j_d_features = get_horse_jockey_distance_features(p, session, prediction)
    h_t_features = get_horse_track_features(p, session, prediction)
    h_j_t_features = get_horse_jockey_track_features(p, session, prediction)
    h_c_features = get_horse_condition_features(p, session, prediction)
    h_j_c_features = get_horse_jockey_condition_features(p, session, prediction)
    j_t_features = get_jockey_trainer_features(p, session, prediction)

    result = np.array(
        p_features +
        horse_features +
        jockey_features +
        trainer_features +
        h_j_features +
        h_d_features +
        j_d_features +
        h_j_d_features +
        h_t_features +
        h_j_t_features +
        h_c_features +
        h_j_c_features +
        j_t_features
    , dtype=np.float32)

    assert result.shape[0] == INDEPENDENT_FEATURES
    return result
