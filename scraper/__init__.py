from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By

from .generate_url import *

import utils.utils as utils
from datetime import datetime
import logging

from dataloader.utils import convert_race_class


def none_if_invalid(
        x,
        invalid_val,
        type_cast=lambda x: x  # default type_cast function is the identity function
):
    if x == invalid_val:
        return None
    else:
        return type_cast(x)


class Scraper:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')

        self.logger = logging.getLogger("Scraper")

        self.driver = webdriver.Chrome(options=options)

    def is_race(self, url: str) -> bool:
        assert url.islower()
        driver = self.driver

        driver.get(url)
        try:
            driver.find_element(By.CLASS_NAME, "top_races")
        except NoSuchElementException:
            return False
        return True

    def scrape_all_horses_urls(self, url: str) -> list[str]:
        driver = self.driver
        driver.get(url)
        table = driver.find_elements(By.CLASS_NAME, "bigborder")[1]
        return list(map(lambda x: x.get_attribute("href").lower(), table.find_elements(By.TAG_NAME, "a")))

    def scrape_horse(self, url: str):
        """
        Read horse from profile in url. First determines whether it is active or retired, then call corrseponding function
        :param url:
        :return:
        """

        # TODO: repeated operation can be reduced to improve efficiency, though not as important for now
        # TODO: lots of repeated code between read_active_horse and read_retired_horse

        assert url.islower()
        driver = self.driver

        try:
            driver.get(url)
            profile = driver.find_element(By.CLASS_NAME, "horseProfile")
            last = profile.find_element(By.CLASS_NAME, "title_text").text.split(" (")[-1]
            if last.rstrip(")") in {"Retired", "Deregistered"}:
                # retired horse
                result = self._read_retired_horse(url)
                result["retired"] = True
            else:
                # active horse
                result = self._read_active_horse(url)
                result["retired"] = False

            # read chinese name
            chi_url = url.replace("english", "chinese")
            driver.get(chi_url)
            profile = driver.find_element(By.CLASS_NAME, "horseProfile")
            name_id = profile.find_element(By.CLASS_NAME, "title_text").text.split(" (")
            name_chi = name_id[0]
            result["name_chi"] = name_chi.strip()

            result["url"] = url
            if "trainer_url" in result:
                trainer_url = convert_trainer_win_stat_to_profile(result["trainer_url"])
                result["trainer_url"] = trainer_url
        except Exception as e:
            e_str = str(e).split("\n")[0]
            self.logger.error(f"Failed to read horse url: {url}. Error: {e_str}")
            return

        return result

    def _read_active_horse(self, url: str) -> dict:
        """
        Read profile of horse only.
        :param url:
        :return:
        """
        assert url.islower()
        driver = self.driver

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

    def _read_retired_horse(self, url: str) -> dict:
        """
        Read profile of retired horse only.
        :param url:
        :return:
        """
        assert url.islower()
        driver = self.driver

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
        result["trainer_name"] = trainer.text
        try:
            result["trainer_url"] = trainer.find_element(By.TAG_NAME, "a").get_attribute("href")
        except (ValueError, NoSuchElementException):
            pass

        # TODO: find way to get age
        result["age"] = None

        return result

    def scrape_num_races(self, url):
        assert url.islower()
        driver = self.driver

        driver.get(url)
        top_races = driver.find_element(By.CLASS_NAME, "top_races")
        return len(top_races.find_element(By.TAG_NAME, "tbody").find_element(By.TAG_NAME, "tr").
                   find_elements(By.TAG_NAME, "td")) - 2

    def scrape_race(self, url: str):
        """
        Read race static details only.
        :param url:
        :return:
        """

        assert url.islower()
        driver = self.driver
        driver.get(url)

        try:
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

            # winnings
            winnings_table = (driver.find_element(By.CLASS_NAME, "localResults")
                              .find_element(By.CLASS_NAME, "dividend_tab")
                              .find_element(By.TAG_NAME, "tbody"))

            rows = winnings_table.find_elements(By.TAG_NAME, "tr")
            current_pool = None
            winnings = []
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) == 2:
                    winning_combination = cells[0].text
                    amount = cells[1].text
                elif len(cells) == 3:
                    current_pool = cells[0].text.strip()
                    winning_combination = cells[1].text
                    amount = cells[2].text
                else:
                    raise Exception("The number of cells in this winnings table row is not 2 or 3")

                if current_pool not in {"WIN", "PLACE", "QUINELLA", "QUINELLA PLACE", "FORECAST", "TIERCE", "TRIO",
                                        "FIRST 4", "QUARTET"}:
                    continue

                amount = amount.replace(",", "")
                if "/$" in amount:
                    payout, unit = amount.split("/$")
                    unit = float(unit)
                    payout = float(payout)
                    amount = payout * (10 / unit)
                else:
                    amount = float(amount)

                winnings.append({
                    "amount": amount,
                    "pool": current_pool,
                    "combination": winning_combination,
                })

            result["winnings"] = winnings
            return result
        except Exception as e:
            error_str = str(e).split("\n")[0]
            self.logger.error(f"Failed to read race url: {url}. Error: {error_str}")
            return None

    def scrape_participation_by_race(self, url):
        assert url.islower()
        driver = self.driver

        # constant info across different horses: race ID and season
        result = []
        driver.get(url)

        try:
            race = driver.find_element(By.CLASS_NAME, "race_tab")
            _, race_id = race.find_element(By.TAG_NAME, "thead").text.split(" (")
            race_id = int(race_id.strip().rstrip(")"))

            race_meeting = driver.find_element(By.CLASS_NAME, "raceMeeting_select")
            _, date, _ = race_meeting.find_element(By.CLASS_NAME, "f_fs13").text.split("  ")
            date = utils.parse_date(date.strip())

            all_participation = (driver.find_element(By.CLASS_NAME, "performance").
                                 find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr"))
        except Exception as e:
            e_str = str(e).split("\n")[0]
            self.logger.error(f"Failed to read participation url: {url}. Error: {e_str}")
            return []

        race_id = utils.build_race_id(race_id, date)
        for idx, p in enumerate(all_participation):
            try:
                cells = p.find_elements(By.TAG_NAME, "td")
                horse_url = cells[2].find_element(By.TAG_NAME, "a").get_attribute("href")
                this_result = {
                    "number": int(cells[1].text),
                    "horse_id": cells[2].text.split(" (")[1].strip().rstrip(")"),
                    "horse_url": horse_url,
                    "race_id": race_id,
                    "ranking": cells[0].text,
                    "lane": none_if_invalid(cells[7].text, "---", int),
                    "gear_weight": none_if_invalid(cells[5].text, "---", int),
                    "horse_weight": none_if_invalid(cells[6].text, "---", int),
                    "finish_time": none_if_invalid(cells[10].text, "---",
                                                   lambda x: datetime.strptime(x, "%M:%S.%f").time()),
                    "win_odds": none_if_invalid(cells[11].text, "---", float),
                }
                try:
                    this_result["jockey_url"] = convert_trainer_win_stat_to_profile(
                        cells[3].find_element(By.TAG_NAME, "a").get_attribute("href"))
                except NoSuchElementException:
                    this_result["jockey_name"] = cells[3].text

                result.append(this_result)
            except Exception as e:
                e_str = str(e).split("\n")[0]
                self.logger.error(f"Failed to read {idx} participation at url: {url}. Error: {e_str}")

        return result

    def scrape_participation_rating(self, url: str, race_id):
        assert url.islower()
        driver = self.driver

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

    def scrape_trainer_jockey(self, url):
        assert url.islower()
        driver = self.driver
        driver.get(url)

        result = {
            "url": url
        }

        lines = driver.find_element(By.TAG_NAME, "table").text.splitlines()
        name = lines[0].strip()
        age_line = lines[1].split(": ")[1].strip()
        try:
            age = int(age_line)
        except ValueError:
            if "-" in age_line:
                age = int(age_line.split(", ")[0].split(" - ")[1].strip())
            elif "–" in age_line:
                age = int(age_line.split(", ")[0].split(" – ")[1].strip())
            else:
                raise Exception

        result["name"] = name
        result["age"] = age

        return result

    def scrape_one_upcoming_race(self, url):

        assert url.islower()
        driver = self.driver

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
            this_p = {"date": date, "location": location, "track": track, "distance": distance,
                      "condition": condition,
                      "total_bet": total_bet, "race_class": race_class}

            cells = row.find_elements(By.TAG_NAME, "td")
            cells = list(filter(lambda x: x.text != "", cells))

            if cells[3].text == "-":
                # Scratched (quit)
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
            this_p["jockey_url"] = convert_trainer_win_stat_to_profile(
                jockey.find_element(By.TAG_NAME, "a").get_attribute("href").lower())
            this_p["trainer_url"] = convert_trainer_win_stat_to_profile(
                trainer.find_element(By.TAG_NAME, "a").get_attribute("href").lower())

            data.append(this_p)

        return data
