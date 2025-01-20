from .config import *
from .utils import *
import numpy as np
from datetime import datetime, timedelta

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
    )


def get_combo_ps(p, ps, jockey=False, trainer=False, condition=False, distance=False, track=False):
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

    return ps


def get_general_group_data(p, ps):
    if len(ps) > 0:
        most_recent = ps[0]
        diff = p.race.date - most_recent.race.date
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
    return [p.rating, p.number, p.lane, p.lane / number_of_participants, p.horse_weight, p.gear_weight]


def load_predict_participation_features(p, number_of_participants):
    return [p["rating"], p["number"], p["lane"], p["lane"] / number_of_participants, p["horse_weight"], p["gear_weight"]]


def load_horse_features(p):
    # length = 48
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days)
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


def load_jockey_features(p):
    # length = 30
    jockey = p.jockey
    ps = jockey.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days)
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


def load_trainer_features(p):
    # length = 19
    trainer = p.horse.trainer
    all_horses = trainer.horses
    ps = []
    for h in all_horses:
        ps += h.participations

    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = ps[:COUNT_RACES_BACKWARD]

    general_group_data = get_general_group_data(p, ps_days)
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


def get_horse_jockey_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, jockey=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def get_horse_distance_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, distance=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def get_jockey_distance_features(p):
    jockey = p.jockey
    ps = jockey.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, distance=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def get_horse_jockey_distance_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, jockey=True, distance=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def get_horse_track_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, track=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def get_horse_jockey_track_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, jockey=True, track=True)

    general_group_data = get_general_group_data(p, ps_days)
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

def get_horse_condition_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, track=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def get_horse_jockey_condition_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, jockey=True, condition=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def get_jockey_trainer_features(p):
    horse = p.horse
    ps = horse.participations
    race_date = p.race.date
    ps = filter_relevant_participations(ps, end_date=race_date)
    ps_days = filter_relevant_participations(ps, end_date=race_date, start_date=race_date - timedelta(days=COUNT_DAYS_BACKWARD))
    ps = get_combo_ps(p, ps, trainer=True)

    general_group_data = get_general_group_data(p, ps_days)
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


def load_one_independent_participation(p):
    p_features = load_participation_features(p)
    horse_features = load_horse_features(p)
    jockey_features = load_jockey_features(p)
    trainer_features = load_trainer_features(p)
    h_j_features = get_horse_jockey_features(p)
    h_d_features = get_horse_distance_features(p)
    j_d_features = get_jockey_distance_features(p)
    h_j_d_features = get_horse_jockey_distance_features(p)
    h_t_features = get_horse_track_features(p)
    h_j_t_features = get_horse_jockey_track_features(p)
    h_c_features = get_horse_condition_features(p)
    h_j_c_features = get_horse_jockey_condition_features(p)
    j_t_features = get_jockey_trainer_features(p)

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


def load_one_predict_independent_participation(session, p, number_of_participants):
    p_features = load_predict_participation_features(p, number_of_participants)


