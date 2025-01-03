from argparse import ArgumentParser

from collect_data import DataCollector
from database import init_engine, get_session
from database.fetch import Fetch
from database.store import Store
from scraper import Scraper, generate_upcoming_race_url

from model import load_model
from model.model_prediction import ModelPrediction

from tabulate import tabulate
from wcwidth import wcswidth


def get_args():
    parser = ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("model_dir")
    parser.add_argument("-db", "--db_path", default="database.db")
    parser.add_argument("race_num", type=int)

    return parser.parse_args()


def get_win_odds(chi_name):
    return 1, 1
    # only function that needs to change when turning into GUI
    individual = float(input(f"{chi_name} 獨贏賠率: "))
    placing = float(input(f"{chi_name} 位置賠率: "))
    return individual, placing


def scrape_one_upcoming_race(data_collector: DataCollector, num):
    fetch = data_collector.fetch
    scraper = data_collector.scraper

    url = generate_upcoming_race_url(num)
    race_data = scraper.scrape_one_upcoming_race(url)

    for ps in race_data:
        horse_url = ps["horse_url"]
        jockey_url = ps["jockey_url"]
        trainer_url = ps["trainer_url"]

        # TODO: handle case where horse, jockey or trainer does not exist
        if not fetch.fetch_horse.exist(url=horse_url):
            data_collector.get_horse(horse_url)

        if not fetch.fetch_jockey.exist(url=jockey_url):
            data_collector.get_jockey(jockey_url)

        if not fetch.fetch_trainer.exist(url=trainer_url):
            data_collector.get_trainer(trainer_url)

        horse_id = fetch.fetch_horse.one(url=horse_url).id
        jockey_id = fetch.fetch_jockey.one(url=jockey_url).id
        trainer_id = fetch.fetch_trainer.one(url=trainer_url).id

        ps.update({"horse_id": horse_id, "jockey_id": jockey_id, "trainer_id": trainer_id})

    return race_data


def main():
    # CAN ONLY BE USED IF TODAY IS A RACE DAY
    args = get_args()

    db_path = "sqlite:///" + args.db_path
    init_engine(db_path)

    model = load_model(args.model_name)
    scraper = Scraper()
    fetch = Fetch()
    store = Store()

    model_predictor = ModelPrediction(model, args.model_dir)
    data_collector = DataCollector(scraper, fetch, store)
    race_data = scrape_one_upcoming_race(data_collector, args.race_num)

    # TODO: consider case where horse does not exist in database
    horse_id = list(map(lambda x: x["horse_id"], race_data))
    horses = list(map(lambda x: fetch.fetch_horse.one(id=x), horse_id))
    chi_names = list(map(lambda x: x.name_chi.strip(), horses))

    win_odds = []
    for idx, data in enumerate(race_data):
        individual, placing = get_win_odds(chi_name=chi_names[idx])
        data["win_odds"] = individual
        win_odds.append((individual, placing))

    session = get_session()
    results = model_predictor(session, race_data)

    model_predictor.display_results(
        horse_id=horse_id,
        chi_names=chi_names,
        results=results,
        win_odds=win_odds,
    )
    session.close()


if __name__ == "__main__":
    main()
