# from datetime import timedelta, datetime
#
# import numpy as np
# import torch
#
# from dataloader.simple_loader import SimpleLoader
# from dataloader.utils import convert_race_class
# from model import load_model
# from database import Participation, fetch, get_session, Race, Horse
#
# import os
#
# import utils.config as config
# from utils import utils
#
#
# def predict(model_name, dataloader, model_dir, data, **kwargs):
#     model = load_model(model_name, dataloader, model_dir, **kwargs)
#     predictions = model(data)
#
#     return predictions
#
#
# if __name__ == '__main__':
#     model_dir = "top_3_nn_epoch_10000/"
#     dataloader = SimpleLoader(0.8)
#     train_mean = np.load(os.path.join(model_dir, "train_mean.npy"))
#     train_std = np.load(os.path.join(model_dir, "train_std.npy"))
#     data = np.array([
#         build_prediction_entry(dataloader, "H202", "Andrea Atzeni", datetime(2024, 12, 11).date(),
#                                "Dennis Yip Chor-hong", 126, 1145, 2.9, 6, 5, 2200, 875000),
#         build_prediction_entry(dataloader, "E448", "Alexis Badel", datetime(2024, 12, 11).date(),
#                                "Chris So Wai-yin", 121, 1035, 4.7, 1, 5, 2200, 875000),
#         build_prediction_entry(dataloader, "G470", "Jerry Chau Chun-lok", datetime(2024, 12, 11).date(),
#                                "Danny Shum Chap-shing", 125, 1225, 21, 7, 5, 2200, 875000),
#         build_prediction_entry(dataloader, "D241", "Karis Teetan", datetime(2024, 12, 11).date(),
#                                "Dennis Yip Chor-hong", 129, 1050, 8.5, 8, 5, 2200, 875000),
#         build_prediction_entry(dataloader, "G172", "Zac Purton", datetime(2024, 12, 11).date(),
#                                "David Hall", 133, 975, 4.1, 2, 5, 2200, 875000),
#         build_prediction_entry(dataloader, "J182", "James McDonald", datetime(2024, 12, 11).date(),
#                                "Pierre Ng Pang Chi", 133, 1228, 10, 3, 5, 2200, 875000),
#         build_prediction_entry(dataloader, "H386", "Alexis Pouchin", datetime(2024, 12, 11).date(),
#                                "Pierre Ng Pang Chi", 132, 1082, 7.3, 5, 5, 2200, 875000),
#         build_prediction_entry(dataloader, "J064", "Ben Thompson", datetime(2024, 12, 11).date(),
#                                "Dennis Yip Chor-hong", 120, 1238, 57, 4, 5, 2200, 875000),
#     ], dtype=np.float32)
#     data, _, _ = dataloader.normalise(data, train_mean, train_std)
#
#     result = predict("Top3NN", dataloader, model_dir, torch.tensor(data, dtype=torch.float32, device=config.device),
#                      input_dim=19)
#     print(result)


from argparse import ArgumentParser

from database import fetch, store
from scraper.scraper import Scraper
from model import load_model

def get_args():
    parser = ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("model_dir")
    parser.add_argument("num_races", type=int)

    return parser.parse_args()


def get_win_odds(race_data):
    # only function that needs to change when turning into GUI
    return float(input(f"Win odds for {race_data['horse_id']}: "))


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
    for race in range(args.num_races):
        race_data = scraper.scrape_one_upcoming_race(race + 1)
        for data in race_data:
            data["win_odds"] = get_win_odds(data)
        print("=" * 50)
        print(model.predict(race_data))
        print("=" * 50)


if __name__ == "__main__":
    main()
