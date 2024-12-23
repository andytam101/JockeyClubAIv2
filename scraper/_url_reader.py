"""
Deals with scraping data from raw urls.
External calls to API should not deal with URLS directly.
"""
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from datetime import datetime

from torch.fx.passes.pass_manager import this_before_that_pass_constraint

from dataloader.utils import convert_race_class
from . import driver
import utils.utils as utils

import calendar

def read_horse(url: str) -> dict:
    """
    Read horse from profile in url. First determines whether it is active or retired, then call corrseponding function
    :param url:
    :return:
    """

    # TODO: repeated operation can be reduced to improve efficiency, though not as important for now
    # TODO: lots of repeated code between read_active_horse and read_retired_horse
    assert url.islower()
    driver.get(url)
    profile = driver.find_element(By.CLASS_NAME, "horseProfile")
    last = profile.find_element(By.CLASS_NAME, "title_text").text.split(" (")[-1]
    if last.rstrip(")") in {"Retired", "Deregistered"}:
        # retired horse
        result = read_retired_horse(url)
        result["retired"] = True
    else:
        # active horse
        result = read_active_horse(url)
        result["retired"] = False

    # read chinese name
    chi_url = url.replace("english", "chinese")
    driver.get(chi_url)
    profile = driver.find_element(By.CLASS_NAME, "horseProfile")
    name_id = profile.find_element(By.CLASS_NAME, "title_text").text.split(" (")
    name_chi = name_id[0]
    result["name_chi"] = name_chi.strip()


    result["url"] = url
    return result


def read_active_horse(url: str) -> dict:
    """
    Read profile of horse only.
    :param url:
    :return:
    """
    assert url.islower()

    driver.get(url)
    result = {}
    profile = driver.find_element(By.CLASS_NAME, "horseProfile")

    eng_name, horse_id = profile.find_element(By.CLASS_NAME, "title_text").text.split(" (")
    eng_name = eng_name.strip()
    horse_id = horse_id.strip().rstrip(")")  # remove closing bracket

    tables = profile.find_elements(By.TAG_NAME, "table")
    left_table = tables[3]
    right_table = tables[4]

    result["name_eng"] = eng_name
    result["id"] = horse_id

    # TODO: refactor this to reduce repeated code
    for row in left_table.text.splitlines():
        try:
            category, data = row.split(" : ")
        except ValueError:
            # TODO: log this error (may be useful)
            continue

        match category:
            case "Country of Origin / Age":
                origin, age = data.split(" / ")
                result["origin"] = origin
                result["age"] = age
            case "Colour / Sex":
                colour_sex = data.split(" / ")
                sex = colour_sex[-1]
                colour = colour_sex[:-1]
                result["colour"] = " / ".join(colour)
                result["sex"] = sex
            case _:
                continue

    # TODO: turn this into for loop (like above) if additional fields are required
    trainer = right_table.find_element(By.TAG_NAME, "tr").find_elements(By.TAG_NAME, "td")[2]
    trainer_name = trainer.text
    result["trainer_name"] = trainer_name
    # give url of trainer in case trainer does not exist in database
    try:
        result["trainer_url"] = trainer.find_element(By.TAG_NAME, "a").get_attribute("href")
    except (NoSuchElementException, ValueError):
        pass

    return result


def read_retired_horse(url: str) -> dict:
    """
    Read profile of retired horse only.
    :param url:
    :return:
    """
    assert url.islower()

    result = {}
    driver.get(url)
    profile = driver.find_element(By.CLASS_NAME, "horseProfile")

    eng_name, horse_id, _ = profile.find_element(By.CLASS_NAME, "title_text").text.split(" (")
    eng_name = eng_name.strip()
    horse_id = horse_id.strip().rstrip(")")  # remove closing bracket

    result["id"] = horse_id
    result["name_eng"] = eng_name

    tables = profile.find_elements(By.TAG_NAME, "table")
    left_table = tables[2]

    for row in left_table.text.splitlines():
        try:
            category, data = row.split(" : ")
        except ValueError:
            # TODO: log this error (may be useful)
            continue

        match category:
            case "Country of Origin":
                origin = data
                result["origin"] = origin
            case "Colour / Sex":
                colour_sex = data.split(" / ")
                sex = colour_sex[-1]
                colour = colour_sex[:-1]
                result["colour"] = " / ".join(colour)
                result["sex"] = sex
            case _:
                continue

    races = driver.find_element(By.CLASS_NAME, "bigborder")
    trainer = races.find_elements(By.TAG_NAME, "tr")[2].find_elements(By.TAG_NAME, "td")[9]
    try:
        result["trainer_url"]  = trainer.find_element(By.TAG_NAME, "a").get_attribute("href")
    except (ValueError, NoSuchElementException):
        result["trainer_name"] = trainer.text

    # TODO: find way to get age
    result["age"] = None

    return result


def read_race(url: str) -> dict:
    """
    Read race static details only.
    :param url:
    :return:
    """
    assert url.islower()

    driver.get(url)

    result = {
        "url": url
    }

    race = driver.find_element(By.CLASS_NAME, "race_tab")

    _, race_id = race.find_element(By.TAG_NAME, "thead").text.split(" (")
    race_id = int(race_id.strip().rstrip(")"))

    result["season_id"] = race_id

    # TODO: refactor into more robust implementation
    rows = race.find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr")
    class_dist, condition = rows[1].text.split("Going : ")
    class_dist = class_dist.split(" - ")
    race_class = class_dist[0].strip()
    distance = int(class_dist[1].strip().rstrip("M"))
    condition = condition.strip()
    total_bet = int(rows[3].text.split()[1].replace(",", ""))

    result["race_class"] = race_class
    result["distance"] = distance
    result["condition"] = condition
    result["total_bet"] = total_bet

    course = rows[2].text.split("Course : ")[1].strip()
    result["course"] = course

    race_meeting = driver.find_element(By.CLASS_NAME, "raceMeeting_select")
    _, date, location = race_meeting.find_element(By.CLASS_NAME, "f_fs13").text.split("  ")

    date = utils.parse_date(date.strip())
    location = location.strip()

    result["date"] = date
    result["location"] = location

    season = utils.calc_season(day=date)
    result["season"] = season

    return result


def read_trainer_jockey(url: str) -> dict:
    """
    Read profile of jockey only.
    :param url:
    :return:
    """
    assert url.islower()

    driver.get(url)

    result = {
        "url": url
    }

    lines = driver.find_element(By.TAG_NAME, "table").text.splitlines()
    name = lines[0].strip()
    age = int(lines[1].split(": ")[1].strip())

    result["name"] = name
    result["age"] = age

    return result


def read_training(url: str) -> list[dict]:
    """
    Record all training records of horse.
    :param url:
    :return:
    """
    assert url.islower()

    # TODO: problem - storing table into database means repeated fields (check for repeats)
    result = []
    driver.get(url)
    rows = (driver.find_element(By.CLASS_NAME, "table_bd").find_element(By.CSS_SELECTOR, "tbody").
            find_elements(By.TAG_NAME, "tr"))
    for row in rows:
        # TODO: turn each row into dictionary
        pass


def read_participations_by_race(url: str) -> list[dict]:
    """
    Takes in a race url. Reads every single participation in that url and return as a list of dictionaries.
    :param url:
    :return:
    """
    assert url.islower()

    # constant info across different horses: race ID and season
    result = []
    driver.get(url)

    race = driver.find_element(By.CLASS_NAME, "race_tab")
    _, race_id = race.find_element(By.TAG_NAME, "thead").text.split(" (")
    race_id = int(race_id.strip().rstrip(")"))

    race_meeting = driver.find_element(By.CLASS_NAME, "raceMeeting_select")
    _, date, _ = race_meeting.find_element(By.CLASS_NAME, "f_fs13").text.split("  ")
    date = utils.parse_date(date.strip())
    season = utils.calc_season(day=date)

    all_participation = (driver.find_element(By.CLASS_NAME, "performance").
                         find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr"))

    for p in all_participation:
        cells = p.find_elements(By.TAG_NAME, "td")

        this_result = {
            "horse_id": cells[2].text.split(" (")[1].strip().rstrip(")"),
            "horse_url": cells[2].find_element(By.TAG_NAME, "a").get_attribute("href"),
            "race_id": utils.build_race_id(race_id, season),
            "ranking": cells[0].text,
            "rating": None,    # TODO: none for now, find a way to get rating (maybe from horse page)
            "lane": none_if_invalid(cells[7].text, "---", int),
            "gear_weight": none_if_invalid(cells[5].text, "---", int),
            "horse_weight": none_if_invalid(cells[6].text, "---", int),
            "finish_time": none_if_invalid(cells[10].text, "---",
                                           lambda x: datetime.strptime(x, "%M:%S.%f").time()),
            "win_odds": none_if_invalid(cells[11].text, "---", float),
        }
        try:
            this_result["jockey_url"] = cells[3].find_element(By.TAG_NAME, "a").get_attribute("href")
        except NoSuchElementException:
            this_result["jockey_name"] = cells[3].text


        # TODO: jockey_url will get traversed multiple times since there are repeated entries, eliminate these entries
        result.append(this_result)

    return result


def read_participations_by_horse(url: str) -> list[dict]:
    """
    Takes in a horse profile url. Reads every single participation in that url and return as a list of dictionaries.
    :param url:
    :return:
    """

    # CLASS: bigborder (html table)


def read_num_races(url: str) -> int:
    """
    Takes in a race url. Returns number of races on that day.
    :param url:
    :return:
    """
    assert url.islower()
    driver.get(url)
    top_races = driver.find_element(By.CLASS_NAME, "top_races")
    return len(top_races.find_element(By.TAG_NAME, "tbody").find_element(By.TAG_NAME, "tr").
        find_elements(By.TAG_NAME, "td")) - 2


def read_is_race(url: str) -> bool:
    assert url.islower()
    driver.get(url)
    try:
        driver.find_element(By.CLASS_NAME, "top_races")
    except NoSuchElementException:
        return False
    return True


def read_all_horses_urls(url: str) -> list[str]:
    """
    Takes in url for base horse
    :param url:
    :return:
    """
    driver.get(url)
    table = driver.find_elements(By.CLASS_NAME, "bigborder")[1]
    return list(map(lambda x: x.get_attribute("href").lower(), table.find_elements(By.TAG_NAME, "a")))


def read_one_upcoming_race(url):
    assert url.islower()
    driver.get(url)
    data = []

    body = driver.find_element(By.TAG_NAME, "body")

    # RACE SPECIFIC THINGS
    race_info = body.find_element(By.CLASS_NAME, "margin_top10").text.split("\n")

    date_location = race_info[1]
    _, month_day, year, location, time = date_location.split(", ")
    month, day = month_day.split()
    date = datetime(int(year), datetime.strptime(month, "%B").month, int(day)).date()

    racing_info = race_info[2].split(", ")
    track = " ".join(racing_info[:-3])
    distance = racing_info[-2]
    condition = racing_info[-1]
    distance = int(distance[:-1])

    prize_class_info = race_info[3].split(", ")
    total_bet = prize_class_info[0].split(": ")[1][1:].replace(",", "")
    race_class = convert_race_class(prize_class_info[2])

    p_table = body.find_element(By.CLASS_NAME, "draggable").find_element(By.TAG_NAME, "tbody")
    p_rows = p_table.find_elements(By.TAG_NAME, "tr")
    for row in p_rows:
        this_p = {"date": date, "location": location, "track": track, "distance": distance, "condition": condition,
                  "total_bet": total_bet, "race_class": race_class}

        cells = row.find_elements(By.TAG_NAME, "td")
        cells = list(filter(lambda x: x.text != "", cells))

        if cells[3].text == "-":
            # Scratched from race (quit)
            continue

        horse = cells[2]
        gear_weight = int(cells[3].text)
        jockey = cells[4]
        lane = int(cells[5].text)
        trainer = cells[6]
        rating = int(cells[7].text)
        horse_weight = int(cells[9].text)

        this_p["gear_weight"] = gear_weight
        this_p["horse_weight"] = horse_weight
        this_p["rating"] = rating
        this_p["lane"] = lane
        this_p["horse_url"] = horse.find_element(By.TAG_NAME, "a").get_attribute("href").lower()
        this_p["jockey_url"] = jockey.find_element(By.TAG_NAME, "a").get_attribute("href").lower()
        this_p["trainer_url"] = trainer.find_element(By.TAG_NAME, "a").get_attribute("href").lower()

        data.append(this_p)

    return data

def none_if_invalid(
        x,
        invalid_val,
        type_cast=lambda x: x   # default type_cast function is the identity function
):
    if x == invalid_val:
        return None
    else:
        return type_cast(x)
