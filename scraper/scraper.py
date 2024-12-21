from lib2to3.pytree import convert

from ._url_reader import *
from ._generate_url import *
from datetime import datetime, timedelta
from tqdm import tqdm

from database import init_engine

import logging


# REPEATED CODE: CREATE SEPARATE FUNCTIONS FOR READING ONE URL (CAUSE HAVE TO HANDLE IF
# DEPENDING-FIELDS DO NOT EXIST, IE A TRAINER FOR A HORSE)
# THEREFORE CREATE A SEPARATE FUNCTION FOR EACH SCRAPE_URL

# TODO: add more robust logging methods (both logging errors and updating latest day scraped)

class Scraper:
    def __init__(
            self,
            store_horse,
            store_race,
            store_participation,
            store_jockey,
            store_trainer,
            store_training,
            fetch_horse,
            fetch_race,
            fetch_participation,
            fetch_jockey,
            fetch_trainer,
            fetch_training
    ):
        init_engine()

        self.store_horse = store_horse
        self.store_race = store_race
        self.store_participation = store_participation
        self.store_jockey = store_jockey
        self.store_trainer = store_trainer
        self.store_training = store_training

        self.fetch_horse = fetch_horse
        self.fetch_race = fetch_race
        self.fetch_participation = fetch_participation
        self.fetch_jockey = fetch_jockey
        self.fetch_trainer = fetch_trainer
        self.fetch_training = fetch_training

        # TODO: get rid of all "magic" stuff
        self.horse_locations = ["HK", "CH"]

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename="scraper.log", level=logging.INFO)

    def get_horse(self, url):
        # TODO: error handling
        horse = read_horse(url)
        if "trainer_url" in horse:
            trainer_url = convert_trainer_jockey_win_stat_to_profile(horse["trainer_url"])
            if not self.fetch_trainer.exist(url=trainer_url):
                self.get_trainer(trainer_url)
            trainer_id = self.fetch_trainer.one(url=trainer_url).id
        else:
            self.store_trainer({
                "name": horse["trainer_name"],
                "age": None,
                "url": None
            })
            trainer_id = self.fetch_trainer.one(name=horse["trainer_name"]).id

        horse["trainer_id"] = trainer_id
        self.store_horse(horse)

    def get_race(self, url):
        try:
            self.store_race(read_race(url))
        except Exception as e:
            self.logger.error(f"Exception occurred: {str(e).splitlines()[0]}, url={url}")

    def get_participation_by_race(self, url):
        participation = read_participations_by_race(url)
        for p in participation:
            if not self.fetch_horse.exist(id=p["horse_id"]):
                self.get_horse(p["horse_url"].lower())

            if "jockey_url" in p:
                jockey_url = convert_trainer_jockey_win_stat_to_profile(p["jockey_url"])
                if not self.fetch_jockey.exist(url=jockey_url):
                    self.get_jockey(jockey_url)
                jockey_id = self.fetch_jockey.filter(url=jockey_url)[0].id
            else:
                self.store_jockey({
                    "name": p["jockey_name"],
                    "age": None,
                    "url": None
                })
                jockey_id = self.fetch_jockey.filter(name=p["jockey_name"])[0].id

            p["jockey_id"] = jockey_id
            self.store_participation(p)

    def get_jockey(self, url):
        jockey = read_trainer_jockey(url)
        self.store_jockey(jockey)

    def get_trainer(self, url):
        trainer = read_trainer_jockey(url)
        self.store_trainer(trainer)

    def get_training(self, url):
        pass

    def get_all_horses(self):
        for location in self.horse_locations:
            print(f"Getting all horses' url from location: {location}")
            url = generate_all_horse_url(location)

            all_urls = read_all_horses_urls(url)
            print(f"Found {len(all_urls)} urls for location: {location}")
            for i in tqdm(all_urls, desc="Reading horses"):
                try:
                    self.get_horse(i)
                except Exception as e:
                    self.logger.error(f"Exception occurred: {str(e).splitlines()[0]}, url={i}")

    def get_all_races_on_date(self, day):
        all_urls = generate_all_race_urls(day)
        for url in all_urls:
            self.get_race(url)

    def get_all_participation_on_date(self, day):
        all_urls = generate_all_race_urls(day)
        for url in all_urls:
            try:
                self.get_participation_by_race(url)
            except Exception as e:
                # if one entry fails in participation, log it down and no other entries get stored
                self.logger.error(f"Exception occurred: {str(e).splitlines()[0]}, url={url}")

    def reload_all_horses(self):
        for url in map(lambda x: x.url, self.fetch_horse.all()):
            self.get_horse(url)

    def get_all_races_from_date(self, start_day):
        today = datetime.now().date()
        for i in range((today - start_day).days):
            day = start_day + timedelta(days=i)
            try:
                if not read_is_race(generate_race_url(day)):
                    self.logger.info(f"Day {day} is not a race")
                    print(f"Day {day} is not a race")
                    continue
                self.logger.info(f"Reading races on day: {datetime.strftime(day, '%d/%m/%Y')}")
                print(f"Reading races on day: {datetime.strftime(day, '%d/%m/%Y')}")
                self.get_all_races_on_date(day)
            except Exception as e:
                self.logger.error(f"Exception on day {day}: {e}")

    def get_all_participation_from_races(self):
        races_url = list(map(lambda x: x.url, self.fetch_race.all()))
        print(f"Found {len(races_url)} urls for races")
        for url in tqdm(races_url, desc="Reading races"):
            try:
                self.get_participation_by_race(url)
            except Exception as e:
                # if one entry fails in participation, log it down and no other entries get stored
                self.logger.error(f"Exception occurred: {str(e).splitlines()[0]}, url={url}")

    def scrape_yesterday(self):
        """
        Call this function at regular intervals.
        :return:
        """
        yesterday = datetime.now() - timedelta(days=1)
        if not read_is_race(generate_race_url(yesterday)):
            return False
        self.get_all_races_on_date(yesterday)
        self.get_all_participation_on_date(yesterday)
        return True

    def scrape_all(self):
        """
        Call this function at the start when new database is initialised.
        :return:
        """
        print("Getting all horses")
        self.get_all_horses()
        print("Getting all races")
        self.get_all_races_from_date(datetime(2022, 9, 1).date())
        print("Getting all participation")
        self.get_all_participation_from_races()

    def update_horses(self):
        horses = self.fetch_horse.all()
        urls = list(map(lambda x: x.url, horses))
        print(f"Found {len(urls)} horses")
        for h in tqdm(urls):
            self.get_horse(h)
