from database import Horse, Participation, Race
from datetime import datetime, time
from utils import utils


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