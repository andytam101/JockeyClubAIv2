"""
Deals with scraping loaded_data from raw urls.
External calls to API should not deal with URLS directly.
"""
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
from datetime import datetime

from dataloader.utils import convert_race_class
from . import driver
import utils.utils as utils

import calendar




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

    all_participation = (driver.find_element(By.CLASS_NAME, "performance").
                         find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr"))

    for p in all_participation:
        cells = p.find_elements(By.TAG_NAME, "td")

        this_result = {
            "horse_id": cells[2].text.split(" (")[1].strip().rstrip(")"),
            "horse_url": cells[2].find_element(By.TAG_NAME, "a").get_attribute("href"),
            "race_id": utils.build_race_id(race_id, date),
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


def read_participation_rating_by_horse(url: str, race_id):
    """
    Takes in a horse profile url. Reads every single participation in that url and return as a list of dictionaries.
    :param url:
    :param race_id:
    :return:
    """
    assert url.islower()
    driver.get(url)

    race_table = driver.find_element(By.CLASS_NAME, "bigborder")
    entries = race_table.find_elements(By.TAG_NAME, "tr")
    for entry in entries:
        try:
            if not entry.text[0].isdigit():
                continue
        except IndexError:
            continue

        cells = entry.find_elements(By.TAG_NAME, "td")
        season_id = int(cells[0].text)
        try:
            date = datetime.strptime(cells[2].text, "%d/%m/%y").date()
        except ValueError:
            date = datetime.strptime(cells[2].text, "%d/%m/%Y").date()
        if utils.build_race_id(season_id, date) != race_id:
            continue
        return none_if_invalid(cells[8].text, "--", int)


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

