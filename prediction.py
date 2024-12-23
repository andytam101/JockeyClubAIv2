from argparse import ArgumentParser
from tabulate import tabulate

from database import fetch, store
from scraper.scraper import Scraper
from model import load_model
from tabulate import tabulate
from wcwidth import wcswidth

def pad_chinese(text, width):
    """Pad Chinese text to ensure it aligns properly."""
    current_width = wcswidth(text)
    return text + " " * (width - current_width)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("model_dir")
    parser.add_argument("race_num", type=int)

    return parser.parse_args()


def get_win_odds(chi_name):
    # only function that needs to change when turning into GUI
    individual = float(input(f"{chi_name} 獨贏賠率: "))
    placing = float(input(f"{chi_name} 位置賠率: "))
    return individual, placing

def main():
    # CAN ONLY BE USED IF TODAY IS A RACE DAY
    args = get_args()
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
    model = load_model(args.model_name, None, args.model_dir)
    race_data = scraper.scrape_one_upcoming_race(args.race_num)
    horse_id = list(map(lambda x: x["horse_id"], race_data))
    horses = list(map(lambda x: fetch.FetchHorse.one(id=x), horse_id))
    chi_names = list(map(lambda x: x.name_chi.strip(), horses))
    win_odds = []
    for idx, data in enumerate(race_data):
        individual, placing = get_win_odds(chi_name=chi_names[idx])
        data["win_odds"] = individual
        win_odds.append(placing)

    results = model.predict(race_data).flatten()
    results_list = results.tolist()
    multiplied = list(map(lambda x: x[0] * x[1], zip(results_list, win_odds)))

    desired_width = max(wcswidth(name) for name in chi_names)
    results_list = [f"{(pred * 100):.1f}%" for pred in results_list]
    win_odds = [f"{odd:.2f}" for odd in win_odds]
    multiplied = [f"{val:.3f}" for val in multiplied]
    chi_names = [pad_chinese(name, desired_width) for name in chi_names]

    # Determine the desired width for the Chinese names
    res = list(map(list, zip(horse_id, chi_names, win_odds, results_list, multiplied)))
    res.sort(key=lambda x: x[4], reverse=True)
    print(tabulate(
        res,
        headers=["編號", "名字", "Place 賠率", "入頭3 機會率", "預期值"],
        tablefmt="psql",
        colalign=["center"] * len(res[0]),
    ))


if __name__ == "__main__":
    main()
