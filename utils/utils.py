from datetime import datetime
from wcwidth import wcswidth
import numpy as np


def calc_season(day=datetime.now()):
    # TODO: get rid of magic number, consider using some dynamic form of way of storing turning date/month
    season = day.year + (day.month >= 9)
    return season


def parse_date(date_str):
    return datetime.strptime(date_str, '%d/%m/%Y').date()


def build_race_id(season_id, date):
    return f"{calc_season(date)}:{int(season_id)}"


def remove_unranked_participants(ps):
    return list(filter(lambda x: x.ranking.replace("DH", "").strip().isnumeric() and x.horse is not None, ps))


def pad_chinese(text, width):
    """Pad Chinese text to ensure it aligns properly."""
    current_width = wcswidth(text)
    return text + " " * (width - current_width)


def randomize_odds(actual_odds):
    upper_bound = actual_odds * 1.25
    lower_bound = actual_odds * 0.75
    return float(np.random.uniform(upper_bound, lower_bound, 1)[0])

def get_track_width(location, track):
    if track == "ALL WEATHER TRACK":
        return SHA_TIN_TRACK_WIDTH[track]

    letter = track.split('"')[1]  # get middle element
    if location == "Sha Tin":
        return SHA_TIN_TRACK_WIDTH[letter]
    elif location == "Happy Valley":
        pass
    else:
        raise Exception(f"Unknown location: {location}")


MAX_NUMBER_OF_HORSES = 14

RACE_LOWER_LIMIT = [100, 80, 60, 40, 0]
RACE_UPPER_LIMIT = [150, 100, 80, 60, 40]

SHA_TIN_TRACK_WIDTH = {
    "A": 30.5,
    "A+2": 28.5,
    "A+3": 27.5,
    "B": 26,
    "B+2": 24,
    "C": 21.3,
    "C+3": 18.3,
    "ALL WEATHER TRACK": 22.8
}

HV_TRACK_WIDTH = {
    "A": 30.5,
    "A+2": 28.5,
    "B": 26.5,
    "B+2": 24.5,
    "C": 22.5,
    "C+3": 19.5,
}

