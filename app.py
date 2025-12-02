from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import numpy as np

from collect_data import DataCollector
from load_data import FinalDataLoader
from pw_models import PWWinnerBinary, PWPlaceBinary, PWRankingScore, PWRelativeRanking
from prediction import load_model, load_data, get_overall_mean_std, scrape_one_upcoming_race, predict_pw, \
    convert_to_x_from_data
from listwise_win_place import ListwiseWinPlace, build_race_x
from true_listwise import ListwiseModel, build_new_race_x, build_listwise_race_x, aggregate_scores, get_mean_std, build_new_x

from database import init_engine, fetch, store
from scraper import Scraper
from utils.config import device
from datetime import datetime


app = Flask(__name__)
CORS(app)
CORS(app, origins="*")


def build_url(date, location, number):
    date_str = datetime.strftime(date, "%Y/%m/%d")
    return f"https://racing.hkjc.com/racing/information/english/racing/RaceCard.aspx?RaceDate={date_str}&Racecourse={location}&RaceNo={number}".lower()


def get_models(model_names):
    win_bin = load_model(PWWinnerBinary, f"final_trained_models/{model_names}/Winner_Binary.pth")
    place_bin = load_model(PWPlaceBinary, f"final_trained_models/{model_names}/Place_Binary.pth")
    ranking_score = load_model(PWRankingScore, f"final_trained_models/{model_names}/Ranking_Score.pth")
    relative_ranking = load_model(PWRelativeRanking, f"final_trained_models/{model_names}/Relative_Ranking.pth")

    listwise_win = load_model(lambda: ListwiseWinPlace(n=6),
                              f"final_trained_listwise/{model_names}/win/model_params.pt")
    listwise_place = load_model(lambda: ListwiseWinPlace(n=6),
                                f"final_trained_listwise/{model_names}/place/model_params.pt")

    pw_models = [win_bin, place_bin, ranking_score, relative_ranking]

    all_data_x, _, _, _, _ = load_data(f"final_loaded_data/{model_names}/weighed/train")
    mean, std = get_overall_mean_std(all_data_x)

    return pw_models, [listwise_win, listwise_place], mean, std


init_engine()
scraper = Scraper()
fetch_api = fetch.Fetch()
store_api = store.Store()
data_collector = DataCollector(scraper, fetch_api, store_api)

dataloader = FinalDataLoader()
dataloader.scale_data = False
dataloader.setup()


@app.route("/prediction")
def prediction():
    number = request.args["number"]
    loc = request.args["loc"]
    dist = request.args["dist"]
    date = request.args["date"]
    date = datetime.strptime(date, "%Y-%m-%d")
    url = build_url(date, loc, number)

    race_data = scrape_one_upcoming_race(data_collector, url)
    pw_models, [listwise_win, listwise_place], mean, std = get_models(f"location_{loc}_{dist}")

    data_x, result_nums = convert_to_x_from_data(race_data, dataloader, data_collector)
    pw_outputs = predict_pw(data_x, mean, std, pw_models)
    list_x, top_n_indices = build_race_x(pw_outputs, n=6)
    listwise_win_output = listwise_win(list_x)
    listwise_place_output = listwise_place(list_x)

    listwise_win.eval()
    listwise_place.eval()

    result_nums = torch.tensor(result_nums, device=device, dtype=torch.int)
    top_n = result_nums[top_n_indices]

    return jsonify({
        "lw_win": listwise_win_output.item(),
        "lw_place": listwise_place_output.item(),
        "top_n": top_n.tolist(),
        "result_nums": result_nums.tolist(),
        "pw_outputs": pw_outputs.tolist()
    })


def get_listwise_model(n, loc, dist):
    model_param = torch.load(f"final_true_listwise_models/location_{loc}_{dist}/n_{n}/model_params.pth", map_location=device)
    model = ListwiseModel(n)
    model.load_state_dict(model_param)
    return model


@app.route("/listwise")
def listwise():
    number = request.args["number"]
    loc = request.args["loc"]
    dist = request.args["dist"]
    date = request.args["date"]
    n = int(request.args["n"])
    win_odds = request.args["win_odds"]
    win_odds = [float(x) for x in win_odds.split(",")]
    date = datetime.strptime(date, "%Y-%m-%d")
    url = build_url(date, loc, number)

    win_odds = torch.tensor(win_odds, dtype=torch.float32, device=device)

    path_name = f"location_{loc}_{dist}"
    races_x = np.load(f"final_loaded_data/{path_name}/weighed/train/data_x.npz")
    races_y = np.load(f"final_loaded_data/{path_name}/weighed/train/data_y.npz")
    data_x = build_new_x(races_x, races_y)
    mean, std = get_mean_std(data_x)

    race_data = scrape_one_upcoming_race(data_collector, url)
    model = get_listwise_model(n, loc, dist)
    race_x, result_nums = convert_to_x_from_data(race_data, dataloader, data_collector)
    race_x = build_new_race_x(race_x, None, win_odds)
    listwise_x = build_listwise_race_x(race_x, mean, std, n)

    model.eval()
    listwise_prediction = model(listwise_x)
    aggregated = aggregate_scores(len(race_x), listwise_prediction, n)

    return jsonify({
        "aggregated": aggregated.tolist(),
        "result_nums": result_nums
    })


@app.route("/group")
def group():
    number = request.args["number"]
    loc = request.args["loc"]
    dist = request.args["dist"]
    date = request.args["date"]
    n = int(request.args["n"])
    horse_nums = request.args["horse_nums"]
    win_odds = request.args["win_odds"]
    horse_nums = [int(x) for x in horse_nums.split(",")]
    win_odds = [float(x) for x in win_odds.split(",")]
    date = datetime.strptime(date, "%Y-%m-%d")
    url = build_url(date, loc, number)

    win_odds = torch.tensor(win_odds, dtype=torch.float32, device=device)
    horse_nums = torch.tensor(horse_nums, dtype=torch.int32, device=device)

    path_name = f"location_{loc}_{dist}"
    races_x = np.load(f"final_loaded_data/{path_name}/weighed/train/data_x.npz")
    races_y = np.load(f"final_loaded_data/{path_name}/weighed/train/data_y.npz")
    data_x = build_new_x(races_x, races_y)
    mean, std = get_mean_std(data_x)

    race_data = scrape_one_upcoming_race(data_collector, url)
    model = get_listwise_model(n, loc, dist)
    race_x, result_nums = convert_to_x_from_data(race_data, dataloader, data_collector)

    result_nums = torch.tensor(result_nums, dtype=torch.int32, device=device)

    mask = torch.isin(result_nums, horse_nums)
    indices = torch.nonzero(mask).flatten()

    race_x = race_x[indices]
    print(indices)
    print(race_x)

    race_x = build_new_race_x(race_x, None, win_odds)
    listwise_x = build_listwise_race_x(race_x, mean, std, n)

    model.eval()
    listwise_prediction = model(listwise_x)
    aggregated = aggregate_scores(len(race_x), listwise_prediction, n)

    assert torch.all(result_nums[indices] == horse_nums)

    return jsonify({
        "aggregated": aggregated.tolist(),
        "result_nums": result_nums[indices].tolist(),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0")
