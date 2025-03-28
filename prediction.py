import torch

from argparse import ArgumentParser

from collect_data import DataCollector
from scraper import Scraper
from database import fetch, store, init_engine, Participation, Race
from scraper.generate_url import generate_upcoming_race_url
from scraper.utils import extract_jockey_trainer_id_from_url
from final_dataloader import FinalDataLoader

from datetime import datetime
from utils.config import device
from tqdm import tqdm

from final_model_analysis import get_overall_mean_std, load_data
from final_models import PWRankingScore, PWRelativeRanking, PWWinnerBinary, PWPlaceBinary

from tabulate import tabulate


def build_upcoming_url(date, location, num):
    return f"https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx?RaceDate={date.strftime("%Y/%m/%d")}&Racecourse={location}&RaceNo={num}".lower()


def get_date_location_max_num():
    date = input("Date: ")
    location = input("Location: ")
    max_num = int(input("Max number: "))

    return datetime.strptime(date, "%Y/%m/%d"), location, max_num


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", type=str)

    return parser.parse_args()


def scrape_one_upcoming_race(data_collector: DataCollector, url):
    fetch_api = data_collector.fetch
    scraper = data_collector.scraper

    race_data = scraper.scrape_one_upcoming_race(url)

    for ps in race_data:
        horse_url = ps["horse_url"]
        jockey_url = ps["jockey_url"]
        trainer_url = ps["trainer_url"]

        jockey_id = extract_jockey_trainer_id_from_url(jockey_url)
        trainer_id = extract_jockey_trainer_id_from_url(trainer_url)

        if not fetch_api.fetch_horse.exist(url=horse_url):
            data_collector.get_horse(horse_url)

        if not fetch_api.fetch_jockey.exist(id=jockey_id):
            data_collector.get_jockey(jockey_url)
            jockey_id = fetch_api.fetch_jockey.one(url=jockey_url).id

        if not fetch_api.fetch_trainer.exist(id=trainer_id):
            data_collector.get_trainer(trainer_url)
            trainer_id = fetch_api.fetch_trainer.one(url=trainer_url).id

        horse_id = fetch_api.fetch_horse.one(url=horse_url).id

        ps.update({"horse_id": horse_id, "jockey_id": jockey_id, "trainer_id": trainer_id})

    return race_data


def group_into_participation(data):
    race = Race(
        date = data["date"],
        distance = data["distance"],
        course = data["course"],
        location = data["location"]
    )
    participation = Participation(
        horse_id = data["horse_id"],
        jockey_id = data["jockey_id"],
        number = data["number"],
        lane = data["lane"],
        rating = data["rating"],
        gear_weight = data["gear_weight"],
        horse_weight = data["horse_weight"],
    )
    participation.race = race

    return participation


def filter_inexperienced(fetch_api, ps):
    result = []
    result_nums = []

    for p in ps:
        horse_id = p["horse_id"]
        ps = fetch_api.fetch_participation(horse_id=horse_id)
        filtered = [p for p in ps if p.finish_time is not None]
        if len(filtered) == 0:
            continue
        result.append(p)
        result_nums.append(p["number"])

    return result, result_nums


def predict_one_race(model, url, data_collector, dataloader, mean, std, display=True):
    fetch_api = data_collector.fetch
    result = scrape_one_upcoming_race(data_collector, url)
    result, result_nums = filter_inexperienced(fetch_api, result)
    data_x = torch.zeros((len(result), 64), dtype=torch.float64, device=device)
    counter = 0

    iterator = result if not display else tqdm(result, desc="Loading data")

    for result_p in iterator:
        this_p = group_into_participation(result_p)
        this_x = dataloader.load_p(this_p, result_p["trainer_id"])
        this_x = torch.tensor(this_x, device=device, dtype=torch.float64)
        data_x[counter] = this_x
        counter += 1

    mean = torch.tensor(mean, dtype=torch.float64, device=device)
    std = torch.tensor(std, dtype=torch.float64, device=device)

    normalized_x = (data_x - mean) / std

    model.eval()
    pred = model(normalized_x)
    pred = pred.flatten().tolist()
    corresponding = list(zip(result_nums, pred))
    corresponding.sort(key=lambda x: x[1], reverse=not model.reverse_points)

    corresponding = [(f"{n}", f"{s:.4f}") for (n, s) in corresponding]

    return corresponding


def get_model(name):
    match name:
        case "PWRScore":
            model = PWRankingScore()
            model_params = torch.load("final_trained_models/Ranking_Score.pth", map_location=device, weights_only=True)
        case "PWRanking":
            model = PWRelativeRanking()
            model_params = torch.load("final_trained_models/Relative_Ranking.pth", map_location=device, weights_only=True)
        case "PWWinBin":
            model = PWWinnerBinary()
            model_params = torch.load("final_trained_models/Winner_Binary.pth", map_location=device, weights_only=True)
        case "PWPlaceBin":
            model = PWPlaceBinary()
            model_params = torch.load("final_trained_models/Place_Binary.pth", map_location=device, weights_only=True)
        case "all":
            return None
        case _:
            raise Exception(f"Unknown model: {name}")

    model.to(device).double()
    model.load_state_dict(model_params)

    return model

def main():
    args = parse_args()
    model = get_model(args.model)
    
    date, location, max_num = get_date_location_max_num()

    init_engine()
    scraper = Scraper()
    fetch_api = fetch.Fetch()
    store_api = store.Store()
    data_collector = DataCollector(scraper, fetch_api, store_api)
    all_data_x, _, _, _, _ = load_data("final_loaded_data/distance_1600/weighed/train")
    mean, std = get_overall_mean_std(all_data_x)
    dataloader = FinalDataLoader()
    dataloader.setup()
    
    for num in range(max_num):
        url = build_upcoming_url(date, location, num + 1)
        if model is not None:
            corresponding = predict_one_race(model, url, data_collector, dataloader, mean, std, display=True)
            print(f"Race {num + 1}")
            print(tabulate(corresponding, headers = ["Horse num", "Score"], tablefmt = "psql"))
        else:
            final_table = []
            print(f"Race {num + 1}")
            for loop_model in ALL_MODELS:
                corresponding = predict_one_race(loop_model, url, data_collector, dataloader, mean, std, display=False)                
                for idx, row in enumerate(corresponding):
                    if len(final_table) <= idx:
                        final_table.append([])
                    final_table[idx].append(row[0])
                    final_table[idx].append(row[1])

            print(tabulate(final_table, headers = ["Num", "RScore", "Num", "WBin", "Num", "PBin", "Num", "Ranking"], tablefmt="psql"))
            

ALL_MODELS = [
    get_model("PWRScore"),
    get_model("PWWinBin"),
    get_model("PWPlaceBin"),
    get_model("PWRanking")
]


if __name__ == '__main__':
    main()
