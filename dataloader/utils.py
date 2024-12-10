from database import init_engine, get_session, Participation, Race
from datetime import datetime, time


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