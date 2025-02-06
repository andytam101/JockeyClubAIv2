from setuptools.command.editable_wheel import editable_wheel

from database import Horse, Participation, Race
from datetime import datetime, time
from utils import utils
import numpy as np
from datetime import timedelta


def calculate_normalized_rankings(p):
    number_of_participants = get_number_of_participants(p.race)
    ranking = get_ranking_from_participation(p)
    return ranking / number_of_participants


def get_number_of_participants(race):
    n = len(utils.remove_unranked_participants(race.participations))
    return n


def get_all_participants(race):
    return utils.remove_unranked_participants(race.participations)


def calculate_speed(p):
    return p.race.distance / time_to_number_of_seconds(p.finish_time)


def get_races_between_dates(session, start=None, end=None):
    races = session.query(Race)
    if start is not None:
        races = races.filter(Race.date >= start)
    if end is not None:
        races = races.filter(Race.date < end)

    return races.all()


def filter_relevant_participations(ps, start_date=None, end_date=None):
    ps = utils.remove_unranked_participants(ps)
    if start_date and end_date:
        ps = list(filter(lambda x: start_date <= x.race.date < end_date, ps))
    elif start_date:
        ps = list(filter(lambda x: start_date <= x.race.date, ps))
    elif end_date:
        ps = list(filter(lambda x: x.race.date < end_date, ps))
    ps.sort(key=lambda x: x.race.date, reverse=True)
    return ps


def mean_std_latest(xs):
    if len(xs) == 0:
        return 0, 0, 0
    return np.mean(xs), np.std(xs), xs[0]


def calculate_running_average(xs, length):
    result = []
    for i in range(length, len(xs) + 1):
        result.append(np.mean(xs[i - length:i]))
    return result

def calculate_gradient(xs):
    return np.diff(xs)


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
    elif words[0] == "Hong" and words[1] == "Kong" and words[2] == "Group":
        return english_to_int(words[3]) / 10
    else:
        return 6
    # elif words[0] == "Griffin" or x == "4 Year Olds":
    #     return 6
    # else:
    #     return 0


def english_to_int(num):
    return ["zero", "one", "two", "three", "four", "five"].index(num.lower())


def time_to_number_of_seconds(t: time):
    return t.second + t.minute * 60 + t.hour * 3600 + t.microsecond * 1e-6


def encode_condition(condition):
    ordering = ['HEAVY', 'WET SLOW', 'WET FAST', 'SOFT', 'YIELDING TO SOFT', 'GOOD TO YIELDING', 'GOOD', 'GOOD TO FIRM', 'FAST']
    if condition in ordering:
        return ordering.index(condition) + 1
    else:
        return 0
