# 64 PERFECT FEATURES - NO TURNING BACK. LAST FILE ON DATA LOADING.
import numpy as np

from database import Race, Participation,  init_engine, get_session
from datetime import time
from utils import utils

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
    if valid_count >= 5:
        return result
    else:
        return None


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

        this_winners = []
        this_places = []
        this_x = np.zeros((m, size), dtype=np.float32)
        this_y = np.zeros((m, 4), dtype=np.float32)
        this_horse_nums = np.zeros(m, dtype=np.int8)

        for i, p in enumerate(experienced_ps):
            p_ranking = get_ranking_from_participation(p)
            this_horse_nums[i] = p.number
            this_y[i, 0] = p_ranking
            this_y[i, 1] = get_seconds_from_time(p.finish_time)
            this_y[i, 2] = get_speed(p)
            this_y[i, 3] = p.win_odds

            this_x[i] = load_p(p)

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


def load_p(p: Participation):
    raise NotImplementedError


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
    load_data()
    session.close()
