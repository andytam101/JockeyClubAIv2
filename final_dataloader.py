# 64 PERFECT FEATURES - NO TURNING BACK. LAST FILE ON DATA LOADING.
import numpy as np

from database import Race, Participation
from datetime import time


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


def get_race_difficulty(race: Race) -> float:
    raise NotImplementedError


def get_race_relevance(race: Race, race_date, race_condition, race_location, jockey_id, trainer_id) -> float:
    raise NotImplementedError


def get_speed_factor(race: Race) -> float:
    distance: int = race.distance
    location: str = race.location
    condition: str = race.condition

    # TODO: come up with equation
    raise NotImplementedError


def get_time_relevancy(current_date, old_date, time_decay=0.03) -> float:
    days_diff = (current_date - old_date).days
    return np.exp(-time_decay * days_diff)


def get_adjusted_speed(p: Participation) -> float:
    raw_speed = get_speed(p)
    factor = get_speed_factor(p.race)
    return raw_speed * factor




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
"""