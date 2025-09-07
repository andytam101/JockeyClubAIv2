from flask import Flask, request, jsonify
import torch

from collect_data import DataCollector
from final_dataloader import FinalDataLoader
from final_models import PWWinnerBinary, PWPlaceBinary, PWRankingScore, PWRelativeRanking
from final_prediction import load_model, load_data, get_overall_mean_std, scrape_one_upcoming_race, predict_pw
from listwise_win_place import ListwiseWinPlace, build_race_x

from database import init_engine, fetch, store
from scraper import Scraper
from utils.config import device

app = Flask(__name__)


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
    url = request.args.get("url", "https://racing.hkjc.com/racing/information/english/racing/RaceCard.aspx?").lower()
    loc = request.args["loc"]
    dist = request.args["dist"]
    race_data = scrape_one_upcoming_race(data_collector, url)
    pw_models, [listwise_win, listwise_place], mean, std = get_models(f"location_{loc}_{dist}")

    pw_outputs, result_nums = predict_pw(race_data, dataloader, data_collector, mean, std, pw_models)
    list_x, top_n_indices = build_race_x(pw_outputs, n=6)
    listwise_win_output = listwise_win(list_x)
    listwise_place_output = listwise_place(list_x)

    result_nums = torch.tensor(result_nums, device=device, dtype=torch.int)
    top_n = result_nums[top_n_indices]

    return jsonify({
        "lw_win": listwise_win_output.item(),
        "lw_place": listwise_place_output.item(),
        "top_n": top_n.tolist(),
        "result_nums": result_nums.tolist(),
        "pw_outputs": pw_outputs.tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0")
