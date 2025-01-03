from datetime import datetime


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
    return url.split("&season=")[0]


def generate_upcoming_race_url(num):
    # ASSUME TODAY IS A RACE DAY
    upcoming_race_base_url = "https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx"
    url = upcoming_race_base_url + f"?RaceDate={datetime.strftime(datetime.today(), '%Y/%m/%d')}" + f"&RaceNo={num}"
    return url.lower()
