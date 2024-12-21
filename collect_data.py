from datetime import datetime

from scraper.scraper import Scraper
from database import store, fetch

import logging
from tqdm import tqdm


def fix():
    scraper = Scraper(
        fetch_horse=fetch.FetchHorse,
        fetch_race=fetch.FetchRace,
        fetch_participation=fetch.FetchParticipation,
        fetch_jockey=fetch.FetchJockey,
        fetch_trainer=fetch.FetchTrainer,
        fetch_training=fetch.FetchTraining,
        store_horse=store.store_horse,
        store_race=store.store_race,
        store_participation=store.store_participation,
        store_jockey=store.store_jockey,
        store_trainer=store.store_trainer,
        store_training=store.store_training,
    )

    logger = logging.getLogger(__name__)
    with open("scraper2.log") as f:
        logs = f.read().splitlines()
    logs = list(filter(lambda x: x.startswith("INFO:scraper.scraper:Exception occurred: 'list' object has no attribute 'id'"), logs))
    for log in tqdm(logs):
        url = log.split("url=")[-1]
        try:
            scraper.get_participation_by_race(url)
        except Exception as e:
            logger.error(f"Exception occurred: {str(e).splitlines()[0]}, url={url}")


def main():
    scraper = Scraper(
        fetch_horse=fetch.FetchHorse,
        fetch_race=fetch.FetchRace,
        fetch_participation=fetch.FetchParticipation,
        fetch_jockey=fetch.FetchJockey,
        fetch_trainer=fetch.FetchTrainer,
        fetch_training=fetch.FetchTraining,
        store_horse=store.store_horse,
        store_race=store.store_race,
        store_participation=store.store_participation,
        store_jockey=store.store_jockey,
        store_trainer=store.store_trainer,
        store_training=store.store_training,
    )

    # scraper.scrape_all()
    # scraper.get_participation_by_race("https://racing.hkjc.com/racing/information/english/racing/localresults.aspx?racedate=2022/09/11&raceno=1")
    scraper.update_horses()

if __name__ == '__main__':
    main()
