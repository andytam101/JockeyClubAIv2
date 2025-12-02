import torch

from argparse import ArgumentParser

from collect_data import DataCollector
from listwise_win_place import ListwiseWinPlace
from scraper import Scraper
from database import fetch, store, init_engine, Participation, Race
from scraper.generate_url import generate_upcoming_race_url
from scraper.utils import extract_jockey_trainer_id_from_url
from load_data import FinalDataLoader

from datetime import datetime
from utils.config import device
from tqdm import tqdm
import numpy as np

from model_analysis import get_overall_mean_std
from pw_models import PWRankingScore, PWRelativeRanking, PWWinnerBinary, PWPlaceBinary

from tabulate import tabulate

from build_group import build_one_group
from listwise_win_place import build_race_x


def get_overall_mean_std(data_x):
    total_size = 0
    for key in data_x:
        total_size += data_x[key].shape[0]

    concatenated = np.zeros((total_size, 64), dtype=np.float64)

    counter = 0
    for key in tqdm(data_x):
        this_size = data_x[key].shape[0]
        concatenated[counter: counter + this_size] = data_x[key]
        counter += this_size

    mean = np.mean(concatenated, axis=0)
    std = np.std(concatenated, axis=0)
    std[std == 0] = 1

    return mean, std


def build_upcoming_url():
    return "https://racing.hkjc.com/racing/information/english/racing/RaceCard.aspx?RaceDate=2025/09/10&Racecourse=HV&RaceNo=1".lower()


def get_date_location_max_num():
    date = input("Date: ")
    location = input("Location: ")
    max_num = int(input("Max number: "))

    return datetime.strptime(date, "%Y/%m/%d"), location, max_num


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
        jockey_id = p["jockey_id"]
        ps = fetch_api.fetch_participation(horse_id=horse_id)
        jockey_ps = fetch_api.fetch_participation(jockey_id=jockey_id)
        filtered_horses = [p for p in ps if p.finish_time is not None]
        filtered_jockeys = [p for p in jockey_ps if p.finish_time is not None]
        if len(filtered_horses) == 0 or len(filtered_jockeys) == 0:
            continue
        result.append(p)
        result_nums.append(p["number"])

    return result, result_nums


def load_model(model_init, path):
    model = model_init()
    model.to(device)
    params = torch.load(path, map_location=device)
    model.load_state_dict(params)
    model.eval()
    return model


def convert_to_x_from_data(
    race_data,
    dataloader,
    datacollector
):
    fetch_api = datacollector.fetch
    result, result_nums = filter_inexperienced(fetch_api, race_data)
    data_x = torch.zeros((len(result), 64), dtype=torch.float32, device=device)
    counter = 0

    for result_p in result:
        this_p = group_into_participation(result_p)
        this_x = dataloader.load_p(this_p, result_p["trainer_id"])
        this_x = np.nan_to_num(this_x, nan=0)
        this_x = torch.tensor(this_x, device=device, dtype=torch.float32)
        data_x[counter] = this_x
        counter += 1

    return data_x, result_nums


def predict_pw(
    data_x,
    mean,
    std,
    models
):
    mean = torch.tensor(mean, device=device, dtype=torch.float32)
    std = torch.tensor(std, device=device, dtype=torch.float32)

    pw_outputs = build_one_group(models, data_x, mean, std)

    return pw_outputs


def load_data(path):
    data_x = np.load(f"{path}/data_x.npz")
    data_y = np.load(f"{path}/data_y.npz")
    horse_nums = np.load(f"{path}/horse_nums.npz")
    wins = np.load(f"{path}/wins.npz")
    places = np.load(f"{path}/places.npz")
    return data_x, data_y, horse_nums, wins, places


def main():
    model_names = "location_ST_1200"
    win_bin = load_model(PWWinnerBinary, f"final_trained_models/{model_names}/Winner_Binary.pth")
    place_bin = load_model(PWPlaceBinary, f"final_trained_models/{model_names}/Place_Binary.pth")
    ranking_score = load_model(PWRankingScore, f"final_trained_models/{model_names}/Ranking_Score.pth")
    relative_ranking = load_model(PWRelativeRanking, f"final_trained_models/{model_names}/Relative_Ranking.pth")

    listwise_win = load_model(lambda: ListwiseWinPlace(n=6),
                              f"final_trained_listwise/{model_names}/win/model_params.pt")
    listwise_place = load_model(lambda: ListwiseWinPlace(n=6),
                                f"final_trained_listwise/{model_names}/place/model_params.pt")

    models = [win_bin, place_bin, ranking_score, relative_ranking]

    init_engine()
    scraper = Scraper()
    fetch_api = fetch.Fetch()
    store_api = store.Store()
    data_collector = DataCollector(scraper, fetch_api, store_api)

    all_data_x, _, _, _, _ = load_data(f"final_loaded_data/{model_names}/weighed/train")
    mean, std = get_overall_mean_std(all_data_x)

    url = build_upcoming_url()
    race_data = scrape_one_upcoming_race(data_collector, url)

    dataloader = FinalDataLoader()
    dataloader.scale_data = False
    dataloader.setup()

    data_x, result_nums = convert_to_x_from_data(race_data, dataloader, data_collector)
    pw_outputs = predict_pw(data_x, mean, std, models)
    print(result_nums)
    print(pw_outputs)

    list_x, top_n_indices = build_race_x(pw_outputs, n=6)
    listwise_win_output = listwise_win(list_x)
    listwise_place_output = listwise_place(list_x)

    result_nums = torch.tensor(result_nums, device=device, dtype=torch.int)

    print(result_nums[top_n_indices])
    print(listwise_win_output)
    print(listwise_place_output)


if __name__ == "__main__":
    main()
