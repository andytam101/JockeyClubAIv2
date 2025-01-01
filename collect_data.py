from scraper import Scraper
from scraper.generate_url import *

from database import init_engine
from database.fetch import Fetch
from database.store import Store

from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime, timedelta
import logging


class DataCollector:
    def __init__(self, scraper, fetch, store, race_url_save_path=None, race_url_found=False):
        self.scraper: Scraper = scraper
        self.fetch: Fetch = fetch
        self.store: Store = store

        self.logger = logging.getLogger("DataCollector")

        self.race_url_save_path = "race_urls.txt" if race_url_save_path is None else race_url_save_path
        self.race_url_found = race_url_found

        # MAGIC STUFF
        self.horse_locations = ["HK", "CH"]

    def get_horse(self, url):
        horse = self.scraper.scrape_horse(url)
        if horse is None:
            return

        trainer_url = horse.get("trainer_url")
        trainer_name = horse["trainer_name"]
        if trainer_url:
            trainer_url = trainer_url.lower()
            if not self.fetch.fetch_trainer.exist(url=trainer_url):
                self.get_trainer(trainer_url)
            trainer_id = self.fetch.fetch_trainer.one(url=trainer_url).id
        else:
            if not self.fetch.fetch_trainer.exist(name=trainer_name):
                self.store.store_trainer({
                    "name": trainer_name,
                    "age": None,
                    "url": None
                })
            trainer_id = self.fetch.fetch_trainer.one(name=trainer_name).id

        horse["trainer_id"] = trainer_id
        self.store.store_horse(horse)

    def get_race(self, url):
        race = self.scraper.scrape_race(url)
        if race is None:
            return
        self.store.store_race(race)

    def get_participations(self, url):
        participations = self.scraper.scrape_participation_by_race(url)
        for p in participations:
            # guaranteed to have horses
            horse_id = p["horse_id"]
            horse_url = p["horse_url"].lower()
            if not self.fetch.fetch_horse.exist(id=horse_id):
                self.get_horse(horse_url)

            jockey_url = p.get("jockey_url")
            if jockey_url:
                jockey_url = jockey_url.lower()
                if not self.fetch.fetch_jockey.exist(url=jockey_url):
                    self.get_jockey(jockey_url)
                jockey_id = self.fetch.fetch_jockey.one(url=jockey_url).id
            else:
                jockey_name = p["jockey_name"]
                if not self.fetch.fetch_jockey.exist(name=jockey_name):
                    self.store.store_jockey({
                        "name": jockey_name,
                        "age": None,
                        "url": None
                    })
                jockey_id = self.fetch.fetch_jockey.one(name=jockey_name).id

            p["jockey_id"] = jockey_id
            p["rating"] = self.scraper.scrape_participation_rating(horse_url, p["race_id"])

            self.store.store_participation(p)

    def get_jockey(self, url):
        jockey = self.scraper.scrape_trainer_jockey(url)
        if jockey is None:
            return
        self.store.store_jockey(jockey)

    def get_trainer(self, url):
        trainer = self.scraper.scrape_trainer_jockey(url)
        if trainer is None:
            return
        self.store.store_trainer(trainer)

    def collect_all_horses_at_location(self, location):
        horse_urls = self.scraper.scrape_all_horses_urls(generate_all_horse_url(location))
        for horse in tqdm(horse_urls, desc=f"Reading horses at location {location}"):
            self.get_horse(horse)

    def collect_all_races_from_date(self, start_date, end_date=datetime.now().date()):
        start_date = start_date.date()
        time_diff = end_date - start_date
        all_urls = []
        if self.race_url_found:
            with open(self.race_url_save_path, "r") as f:
                all_urls = f.readlines()
        else:
            for i in tqdm(range(time_diff.days), desc="Building races urls..."):
                current_date = start_date + timedelta(days=i)
                first_url = generate_race_url(current_date, 1)
                if not self.scraper.is_race(first_url):
                    continue
                total_races = self.scraper.scrape_num_races(first_url)
                all_urls += generate_all_race_urls(current_date, total_races)

        with open(self.race_url_save_path, "w") as f:
            for url in all_urls:
                f.write(f"{url}\n")

        print(f"Found {len(all_urls)} races")
        for url in tqdm(all_urls, desc="Reading races"):
            self.get_race(url)

    def collect_all_participations(self):
        all_races_url = self.fetch.fetch_race.all_url()
        for url in tqdm(all_races_url, desc="Reading participations"):
            self.get_participations(url)

    def collect_all_data(self, start_date):
        # read all horses
        print("Reading horses")
        for location in self.horse_locations:
            self.collect_all_horses_at_location(location)

        # read all races
        print("Reading races")
        self.collect_all_races_from_date(start_date)

        # read all participations
        print("Reading participations")
        self.collect_all_participations()

    def collect_yesterday_race(self):
        yesterday = datetime.now().date() - timedelta(days=1)
        num_races = self.scraper.scrape_num_races(yesterday)
        urls = generate_all_race_urls(yesterday, num_races)
        for url in tqdm(urls, desc="Reading yesterday's races"):
            self.get_race(url)

    def update_horses(self):
        all_horses_url = self.fetch.fetch_horse.all_url()
        for url in tqdm(all_horses_url, desc="Updating horses"):
            self.get_horse(url)

    def update_jockeys(self):
        all_jockeys_url = self.fetch.fetch_jockey.all_url()
        for url in tqdm(all_jockeys_url, desc="Updating jockeys"):
            self.get_jockey(url)

    def update_trainers(self):
        all_trainers_url = self.fetch.fetch_trainer.all_url()
        for url in tqdm(all_trainers_url, desc="Updating trainers"):
            self.get_trainer(url)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument("db_path", default="database.db")
    parser.add_argument("-s", "--start_date", default="2022/09/01")
    parser.add_argument("-e", "--end_date")
    parser.add_argument("-p", "--url_path", default=None)
    parser.add_argument("-f", "--url_found", action="store_true")
    parser.add_argument("-l", "--logging", default="collect_data.log")
    return parser.parse_args()


def main():
    args = parse_args()

    # temporary (may have to change to something else other than sqlite in the future)
    db_path = "sqlite:///" + args.db_path
    init_engine(db_path)

    logging.basicConfig(filename=args.logging, level=logging.INFO)

    data_collector = DataCollector(Scraper(), Fetch(), Store(), race_url_save_path=args.url_path,
                                   race_url_found=args.url_found)
    instruction = args.instruction.lower()
    start_date = datetime.strptime(args.start_date, "%Y/%m/%d")
    if instruction in {"build", "b"}:
        data_collector.collect_all_data(start_date)
    elif instruction in {"update", "u"}:
        data_collector.collect_yesterday_race()
        data_collector.update_horses()
        data_collector.update_jockeys()
        data_collector.update_trainers()


if __name__ == "__main__":
    main()
