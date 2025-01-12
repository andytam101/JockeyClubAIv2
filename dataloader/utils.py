from database import Horse, Participation, Race
from datetime import datetime, time
from utils import utils
import numpy as np
from datetime import timedelta

INDIVIDUAL_FEATURES = 45


def get_training_participations(session, start_date=None, end_date=None):
    ps = session.query(Participation).join(Race)

    if start_date is not None:
        ps = ps.filter(Race.date >= start_date)

    if end_date is not None:
        ps = ps.filter(Race.date < end_date)

    ps = ps.all()
    ps = utils.remove_unranked_participants(ps)
    return ps


def get_relevant_participation(session, before, after, horse_id=None, race_id=None, jockey_id=None, trainer_id=None):
    ps = (session.query(Participation).join(Race).join(Horse)
          .filter(Race.date < before)
          .filter(Race.date >= after))

    if horse_id is not None:
        ps = ps.filter(Horse.id == horse_id)
    elif race_id is not None:
        ps = ps.filter(Race.id == race_id)
    elif jockey_id is not None:
        ps = ps.filter(Participation.jockey_id == jockey_id)
    elif trainer_id is not None:
        ps = ps.filter(Horse.trainer_id == trainer_id)

    result = ps.all()
    return utils.remove_unranked_participants(result)


def get_relevant_participations(session, before, after_days_count, horse_id, jockey_id, trainer_id):
    after = before - timedelta(after_days_count)

    horse_ps = get_relevant_participation(session, before, after, horse_id=horse_id)
    jockey_ps = get_relevant_participation(session, before, after, jockey_id=jockey_id)
    trainer_ps = get_relevant_participation(session, before, after, trainer_id=trainer_id)

    return horse_ps, jockey_ps, trainer_ps


def get_participation_before(session, horse_id, day):
    ans = (
        session.query(Participation).join(Race)
        .filter(Participation.horse_id == horse_id)
        .filter(Race.date < day)
        .order_by(Race.date.desc())
        .all()
    )
    return ans


def get_ranking_from_participation(x):
    return int(x.ranking.replace("DH", "").strip())


def convert_race_class(x):
    words = x.split()
    if words[0] == "Class":
        return int(words[1])
    elif words[0] == "Group":
        return english_to_int(words[1]) / 10
    elif words[0] == "Griffin":
        return 6


def english_to_int(num):
    return ["zero", "one", "two", "three", "four", "five"].index(num.lower())


def time_to_number_of_seconds(t: time):
    return t.second + t.minute * 60 + t.hour * 3600 + t.microsecond * 1e-6


def get_participation_data(p: Participation):
    return [
        p.lane,
        p.number,
        p.gear_weight,
        p.rating if p.rating is not None else 25,
        p.horse_weight,
    ]


def get_horse_data(h: Horse):
    return [
        h.age  # TODO: subtract now from race date
    ]


def get_race_data(r: Race):
    return [
        convert_race_class(r.race_class),
        r.distance,
        r.total_bet,
        len(r.participations),
    ]


def get_mean_std(data):
    data = list(filter(lambda x: x is not None, data))
    if len(data) == 0:
        return [0, 0]
    else:
        return [
            np.mean(data),
            np.std(data),
        ]


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


def get_group_speed(ps: list[Participation]):
    if len(ps) == 0:
        return [0, 0]
    speeds = list(map(lambda x: time_to_number_of_seconds(x.finish_time) / x.race.distance, ps))
    return [np.mean(speeds), np.std(speeds)]


def get_group_ranking(ps: list[Participation]):
    if len(ps) == 0:
        return [0, 0]

    ranking = list(map(lambda x: get_ranking_from_participation(x), ps))
    upper_quartile, lower_quartile = np.percentile(ranking, [75, 25])
    return [np.median(ranking), upper_quartile - lower_quartile]


def get_group_horse_weights(ps: list[Participation]):
    if len(ps) == 0:
        return [0, 0]
    weights = list(map(lambda x: x.horse_weight, ps))
    return [np.mean(weights), np.std(weights)]


def get_group_gear_weights(ps: list[Participation]):
    if len(ps) == 0:
        return [0, 0]
    weights = list(map(lambda x: x.gear_weight, ps))
    return [np.mean(weights), np.std(weights)]


def get_group_win_odds(ps: list[Participation]):
    if len(ps) == 0:
        return [0, 0]
    odds = list(map(lambda x: x.horse_weight, ps))
    return [np.mean(odds), np.std(odds)]


def get_group_top_ratio(ps: list[Participation], ranking=1):
    total = len(ps)
    top_count = len(list(filter(lambda x: get_ranking_from_participation(x) <= ranking, ps)))
    if total == 0:
        return 0
    return top_count / total


def get_grouped_stats(ps: list[Participation]):
    # sort participations by racing date
    ps.sort(key=lambda x: x.race.date, reverse=True)

    dates = list(map(lambda x: x.race.date, ps))

    speeds = list(map(lambda x: x.race.distance / time_to_number_of_seconds(x.finish_time), ps))
    speeds_diff, speeds_derivative = (
        calculate_change_and_derivative_by_date(speeds, dates))

    rankings = list(map(lambda x: get_ranking_from_participation(x) / len(
        utils.remove_unranked_participants(x.race.participations)), ps))

    top_1_count = len(list(filter(lambda k: k == 1, rankings)))
    top_2_count = len(list(filter(lambda k: k <= 2, rankings)))
    top_3_count = len(list(filter(lambda k: k <= 3, rankings)))
    top_4_count = len(list(filter(lambda k: k <= 4, rankings)))

    # rating
    ratings = list(map(lambda x: x.rating, ps))
    ratings_diff, ratings_derivative = (
        calculate_change_and_derivative_by_date(ratings, dates)
    )

    # weight changes
    horse_weights = list(map(lambda x: x.horse_weight, ps))
    horse_weights_diff, horse_weights_derivative = (
        calculate_change_and_derivative_by_date(horse_weights, dates)
    )

    # win odds
    win_odds = list(map(lambda x: x.win_odds, ps))
    win_odds_diff, win_odds_derivative = (
        calculate_change_and_derivative_by_date(win_odds, dates)
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

        # latest horse weight
        0 if len(horse_weights) == 0 else horse_weights[0],

        # latest win odds
        0 if len(win_odds) == 0 else win_odds[0],
    ]

    result += get_mean_std(rankings)
    result += get_mean_std(speeds)
    result += get_mean_std(ratings)
    result += get_mean_std(horse_weights)
    result += get_mean_std(win_odds)

    return result


def load_individual_participation(session, p: Participation):
    static_data = (get_participation_data(p)
                   + get_race_data(p.race)
                   )

    horse_ps, jockey_ps, trainer_ps = get_relevant_participations(session,
                                                                  p.race.date, 90, p.horse_id, p.jockey_id,
                                                                  p.horse.trainer_id)

    if len(horse_ps) > 0:
        latest_horse_p = max(horse_ps, key=lambda x: x.race.date).race.date
    else:
        latest_horse_p = p.race.date

    if len(jockey_ps) > 0:
        latest_jockey_p = max(jockey_ps, key=lambda x: x.race.date).race.date
    else:
        latest_jockey_p = p.race.date

    num_days_horse = (p.race.date - latest_horse_p).days
    num_days_jockey = (p.race.date - latest_jockey_p).days

    horse_ps_data = ([len(horse_ps)]
                     + get_group_speed(horse_ps)
                     + get_group_ranking(horse_ps)
                     + get_group_horse_weights(horse_ps)
                     + get_group_win_odds(horse_ps)
                     + [get_group_top_ratio(horse_ps, 1)]
                     + [get_group_top_ratio(horse_ps, 2)]
                     + [get_group_top_ratio(horse_ps, 3)]
                     + [get_group_top_ratio(horse_ps, 4)])

    jockey_ps_data = ([len(jockey_ps)]
                      + get_group_speed(jockey_ps)
                      + get_group_ranking(jockey_ps)
                      + get_group_gear_weights(jockey_ps)
                      + get_group_win_odds(jockey_ps)
                      + [get_group_top_ratio(jockey_ps, 1)]
                      + [get_group_top_ratio(jockey_ps, 2)]
                      + [get_group_top_ratio(jockey_ps, 3)]
                      + [get_group_top_ratio(jockey_ps, 4)])

    trainer_speed_mean, _ = get_group_speed(trainer_ps)
    trainer_ranking_median, _ = get_group_ranking(trainer_ps)
    trainer_win_odds_mean, _ = get_group_win_odds(trainer_ps)
    trainer_top_1_ratio = get_group_top_ratio(trainer_ps, 1)
    trainer_top_2_ratio = get_group_top_ratio(trainer_ps, 2)
    trainer_top_3_ratio = get_group_top_ratio(trainer_ps, 3)
    trainer_top_4_ratio = get_group_top_ratio(trainer_ps, 4)

    trainer_ps_data = [
        len(trainer_ps),
        trainer_speed_mean,
        trainer_ranking_median,
        trainer_win_odds_mean,
        trainer_top_1_ratio,
        trainer_top_2_ratio,
        trainer_top_3_ratio,
        trainer_top_4_ratio
    ]

    result = np.array(static_data + [num_days_horse, num_days_jockey]
                      + horse_ps_data
                      + jockey_ps_data
                      + trainer_ps_data,
                      dtype=np.float32)

    result = np.nan_to_num(result)
    return result


def load_individual_predict(session, **kwargs):
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
    race_class = kwargs["race_class"]  # convert_race_class should have already been called
    distance = kwargs["distance"]
    total_bet = kwargs["total_bet"]
    win_odds = kwargs["win_odds"]
    number_of_horses = kwargs["number_of_horses"]

    entry = np.zeros(INDIVIDUAL_FEATURES, dtype=np.float32)
    entry[0] = lane
    entry[1] = number
    entry[2] = gear_weight
    entry[3] = rating if rating is not None else 25
    entry[4] = horse_weight
    entry[5] = race_class
    entry[6] = distance
    entry[7] = total_bet
    entry[8] = number_of_horses

    horse_ps, jockey_ps, trainer_ps = get_relevant_participations(session,
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

    horse_ps_data = ([len(horse_ps)]
                     + get_group_speed(horse_ps)
                     + get_group_ranking(horse_ps)
                     + get_group_horse_weights(horse_ps)
                     + get_group_win_odds(horse_ps)
                     + [get_group_top_ratio(horse_ps, 1)]
                     + [get_group_top_ratio(horse_ps, 2)]
                     + [get_group_top_ratio(horse_ps, 3)]
                     + [get_group_top_ratio(horse_ps, 4)])

    jockey_ps_data = ([len(jockey_ps)]
                      + get_group_speed(jockey_ps)
                      + get_group_ranking(jockey_ps)
                      + get_group_gear_weights(jockey_ps)
                      + get_group_win_odds(jockey_ps)
                      + [get_group_top_ratio(jockey_ps, 1)]
                      + [get_group_top_ratio(jockey_ps, 2)]
                      + [get_group_top_ratio(jockey_ps, 3)]
                      + [get_group_top_ratio(jockey_ps, 4)])

    trainer_speed_mean, _ = get_group_speed(trainer_ps)
    trainer_ranking_median, _ = get_group_ranking(trainer_ps)
    trainer_win_odds_mean, _ = get_group_win_odds(trainer_ps)
    trainer_top_1_ratio = get_group_top_ratio(trainer_ps, 1)
    trainer_top_2_ratio = get_group_top_ratio(trainer_ps, 2)
    trainer_top_3_ratio = get_group_top_ratio(trainer_ps, 3)
    trainer_top_4_ratio = get_group_top_ratio(trainer_ps, 4)

    trainer_ps_data = [
        len(trainer_ps),
        trainer_speed_mean,
        trainer_ranking_median,
        trainer_win_odds_mean,
        trainer_top_1_ratio,
        trainer_top_2_ratio,
        trainer_top_3_ratio,
        trainer_top_4_ratio
    ]

    entry[11:INDIVIDUAL_FEATURES] = np.array(
        horse_ps_data + jockey_ps_data + trainer_ps_data
    )

    return entry
