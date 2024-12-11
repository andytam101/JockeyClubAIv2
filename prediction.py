from datetime import timedelta

import numpy as np

from dataloader.simple_loader import SimpleLoader
from dataloader.utils import convert_race_class
from model import load_model
from database import Participation, fetch, get_session, Race, Horse

import os


def predict(model_name, model_dir, dataloader, participations, **kwargs):
    model = load_model(model_name, model_dir, **kwargs)
    training_mean = os.path.join(model_dir, 'training_mean.npy')
    training_std = os.path.join(model_dir, 'training_std.npy')

    x_data = dataloader.load_n_participations(participations, training_mean, training_std)
    predictions = model(x_data)

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
        train_mean,
        train_std
):
    jockey = fetch.FetchJockey(name=jockey_name)
    trainer = fetch.FetchTrainer(name=trainer_name)
    entry = np.zeros(19, dtype=np.float32)

    entry[0] = lane
    entry[1] = gear_weight
    entry[2] = horse_weight
    entry[3] = win_odds
    entry[4] = convert_race_class(race_class)
    entry[5] = distance
    entry[6] = total_bet

    # TODO: repeated code in SimpleLoader
    before = date
    after = before - timedelta(days=90)

    session = get_session()
    relevant_ps = (session.query(Participation).join(Race).join(Horse)
                   .filter(Race.date < before)
                   .filter(Race.date >= after)
                   )
    horse_ps = relevant_ps.filter(Participation.horse_id == horse_id).all()
    jockey_ps = relevant_ps.filter(Participation.jockey_id == jockey.id).all()
    trainer_ps = relevant_ps.filter(Horse.trainer_id == trainer.id).all()

    entry[7:11] = np.array(dataloader.get_grouped_stats(horse_ps), dtype=np.float32)
    entry[11:15] = np.array(dataloader.get_grouped_stats(jockey_ps), dtype=np.float32)
    entry[15:19] = np.array(dataloader.get_grouped_stats(trainer_ps), dtype=np.float32)

    session.close()

    entry = dataloader.normalise(entry, train_mean, train_std)
    return entry


if __name__ == '__main__':
    result = predict("Top3NN", "top_3_nn_epoch_100000.pth", [])
