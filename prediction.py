from datetime import timedelta, datetime

import numpy as np
import torch

from dataloader.simple_loader import SimpleLoader
from dataloader.utils import convert_race_class
from model import load_model
from database import Participation, fetch, get_session, Race, Horse

import os

import utils.config as config
from utils import utils


def predict(model_name, dataloader, model_dir, data, **kwargs):
    model = load_model(model_name, dataloader, model_dir, **kwargs)
    predictions = model(data)

    return predictions


def build_prediction_entry(
        dataloader: SimpleLoader,
        horse_id,
        jockey_name,
        date,
        trainer_name,
        gear_weight,
        horse_weight,
        win_odds,
        lane,
        race_class,
        distance,
        total_bet,
):
    entry = np.zeros(19, dtype=np.float32)

    entry[0] = lane
    entry[1] = gear_weight
    entry[2] = horse_weight
    entry[3] = win_odds
    # entry[4] = convert_race_class(race_class)
    entry[4] = race_class
    entry[5] = distance
    entry[6] = total_bet

    # TODO: repeated code in SimpleLoader
    before = date
    after = before - timedelta(days=90)

    jockey = fetch.FetchJockey.filter(name=jockey_name)
    trainer = fetch.FetchTrainer.filter(name=trainer_name)

    session = get_session()
    relevant_ps = (session.query(Participation).join(Race).join(Horse)
                   .filter(Race.date < before)
                   .filter(Race.date >= after)
                   )
    horse_ps = utils.remove_unranked_participants(relevant_ps.filter(Participation.horse_id == horse_id).all())
    if len(jockey) > 0:
        jockey_ps = utils.remove_unranked_participants(
            relevant_ps.filter(Participation.jockey_id == jockey[0].id).all())
    else:
        jockey_ps = []
    if len(trainer) > 0:
        trainer_ps = utils.remove_unranked_participants(relevant_ps.filter(Horse.trainer_id == trainer[0].id).all())
    else:
        trainer_ps = []

    entry[7:11] = np.nan_to_num(np.array(dataloader.get_grouped_stats(horse_ps), dtype=np.float32))
    entry[11:15] = np.nan_to_num(np.array(dataloader.get_grouped_stats(jockey_ps), dtype=np.float32))
    entry[15:19] = np.nan_to_num(np.array(dataloader.get_grouped_stats(trainer_ps), dtype=np.float32))

    session.close()

    return entry


if __name__ == '__main__':
    model_dir = "top_3_nn_epoch_10000/"
    dataloader = SimpleLoader(0.8)
    train_mean = np.load(os.path.join(model_dir, "train_mean.npy"))
    train_std = np.load(os.path.join(model_dir, "train_std.npy"))
    data = np.array([
        build_prediction_entry(dataloader, "H202", "Andrea Atzeni", datetime(2024, 12, 11).date(),
                               "Dennis Yip Chor-hong", 126, 1145, 2.9, 6, 5, 2200, 875000),
        build_prediction_entry(dataloader, "E448", "Alexis Badel", datetime(2024, 12, 11).date(),
                               "Chris So Wai-yin", 121, 1035, 4.7, 1, 5, 2200, 875000),
        build_prediction_entry(dataloader, "G470", "Jerry Chau Chun-lok", datetime(2024, 12, 11).date(),
                               "Danny Shum Chap-shing", 125, 1225, 21, 7, 5, 2200, 875000),
        build_prediction_entry(dataloader, "D241", "Karis Teetan", datetime(2024, 12, 11).date(),
                               "Dennis Yip Chor-hong", 129, 1050, 8.5, 8, 5, 2200, 875000),
        build_prediction_entry(dataloader, "G172", "Zac Purton", datetime(2024, 12, 11).date(),
                               "David Hall", 133, 975, 4.1, 2, 5, 2200, 875000),
        build_prediction_entry(dataloader, "J182", "James McDonald", datetime(2024, 12, 11).date(),
                               "Pierre Ng Pang Chi", 133, 1228, 10, 3, 5, 2200, 875000),
        build_prediction_entry(dataloader, "H386", "Alexis Pouchin", datetime(2024, 12, 11).date(),
                               "Pierre Ng Pang Chi", 132, 1082, 7.3, 5, 5, 2200, 875000),
        build_prediction_entry(dataloader, "J064", "Ben Thompson", datetime(2024, 12, 11).date(),
                               "Dennis Yip Chor-hong", 120, 1238, 57, 4, 5, 2200, 875000),
    ], dtype=np.float32)
    data, _, _ = dataloader.normalise(data, train_mean, train_std)

    result = predict("Top3NN", dataloader, model_dir, torch.tensor(data, dtype=torch.float32, device=config.device),
                     input_dim=19)
    print(result)
