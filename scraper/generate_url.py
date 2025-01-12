from datetime import datetime
from utils.pools import *


def generate_race_url(day, number=1):
    """
    Generates HKJC race url by day and race number.
    :param day:
    :param number:
    :return:
    """
    # TODO: remove "magic string" and store it in a separate file
    race_base = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?"
    day_str = "RaceDate=" + datetime.strftime(day, "%Y/%m/%d")
    num_str = "RaceNo=" + str(number)
    return (race_base + day_str + "&" + num_str).lower()


def generate_all_race_urls(day, total_races) -> list[str]:
    # get number of races
    original_url = generate_race_url(day)

    result = [original_url]
    for i in range(2, total_races + 1):
        result.append(generate_race_url(day, i))

    return result


def generate_all_horse_url(location):
    # TODO: get rid of "magic urls"
    all_horse_base_url = "https://racing.hkjc.com/racing/information/english/Horse/ListByLocation.aspx"
    all_horse_base_url += f"?Location={location}"
    return all_horse_base_url.lower()


def convert_trainer_win_stat_to_profile(url: str):
    url = url.replace("TrainerWinStat", "TrainerProfile").lower()
    return url


def generate_upcoming_race_url(num):
    # ASSUME TODAY IS A RACE DAY
    upcoming_race_base_url = "https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx"
    url = upcoming_race_base_url + f"?RaceDate={datetime.strftime(datetime.today(), '%Y/%m/%d')}" + f"&RaceNo={num}"
    return url.lower()

def convert_win_odds_url(url: str, pool):
    # ensure consistency of url
    assert url.islower()

    # ensure url is a win / place win odds url
    assert "racing/wp" in url

    if pool in {WIN, PLACE}:
        return url
    elif pool in {QUINELLA, Q_PLACE}:
        return url.replace("wp", "wpq")
    elif pool == FORECAST:
        return url.replace("wp", "fct")
    elif pool == TIERCE:
        return url.replace("wp", "tce")
    elif pool == TRIO:
        return url.replace("wp", "tri")
    elif pool == FIRST_4:
        return url.replace("wp", "ff")
    elif pool == QUARTET:
        return url.replace("wp", "ff")
    else:
        raise Exception(f"Unsupported pool: {pool}")
