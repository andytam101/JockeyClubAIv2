from ._url_reader import *
from .generate_url import *
from datetime import datetime, timedelta
from tqdm import tqdm

from database import init_engine

import logging


# REPEATED CODE: CREATE SEPARATE FUNCTIONS FOR READING ONE URL (CAUSE HAVE TO HANDLE IF
# DEPENDING-FIELDS DO NOT EXIST, IE A TRAINER FOR A HORSE)
# THEREFORE CREATE A SEPARATE FUNCTION FOR EACH SCRAPE_URL

# TODO: add more robust logging methods (both logging errors and updating latest day scraped)

class Scraper:
    def __init__(self):
        init_engine()

        # TODO: get rid of all "magic" stuff
        self.horse_locations = ["HK", "CH"]
        self.base_prediction_url = "https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx?"

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename="scraper.log", level=logging.INFO)

    def is_race(self, url: str) -> bool:
        assert url.islower()
        driver.get(url)
        try:
            driver.find_element(By.CLASS_NAME, "top_races")
        except NoSuchElementException:
            return False
        return True

    def scrape_horse(self, url):
        # TODO: error handling
        horse = read_horse(url)
        trainer_url = convert_trainer_win_stat_to_profile(horse["trainer_url"])
        horse["trainer_url"] = trainer_url
        return horse

    def scrape_race(self, url):
        try:
            return read_race(url)
        except Exception as e:
            self.logger.error(f"Exception occurred: {str(e).splitlines()[0]}, url={url}")
            return None

    def scrape_participation_by_race(self, url):
        participation = read_participations_by_race(url)
        for p in participation:
            p["rating"] = read_participation_rating_by_horse(p["horse_url"], p["race_id"])
        return participation

    def scrape_jockey(self, url):
        jockey = read_trainer_jockey(url)
        return jockey

    def scrape_trainer(self, url):
        trainer = read_trainer_jockey(url)
        return trainer

    def scrape_training(self, url):
        pass

    def scrape_one_upcoming_race(self, num):
        # ASSUME TODAY IS A RACE DAY
        url = generate_upcoming_race_url(num)
        race_data = read_one_upcoming_race(url)
        for ps in race_data:
            horse_url = ps["horse_url"]
            jockey_url = ps["jockey_url"]
            trainer_url = ps["trainer_url"]

            # making sure they exist
            self.get_horse(horse_url)
            self.get_jockey(jockey_url)
            self.get_trainer(trainer_url)

            horse_id = self.fetch.fetch_horse.one(url=horse_url).id
            jockey_id = self.fetch.fetch_jockey.one(url=jockey_url).id
            trainer_id = self.fetch.fetch_trainer.one(url=trainer_url).id

            ps.update({"horse_id": horse_id, "jockey_id": jockey_id, "trainer_id": trainer_id})

        return race_data
